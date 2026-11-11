// nn::time: relojes del sistema y conversiones de calendario, sobre la hora de Android. NieR fecha
// las partidas guardadas con ellos (antes eran stubs y la fecha salía sin inicializar).
//
// La zona horaria es la del teléfono (persist.sys.timezone) y las reglas, las de la base de datos de zonas de
// Android: una TimeZoneRule solo lleva el nombre de su zona.

#include <sys/system_properties.h>
#include <time.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "nnsdk/types.h"

namespace nn::time {

struct PosixTime {
    int64_t value;
};
struct CalendarTime {
    int16_t year;
    int8_t month, day, hour, minute, second;
    int8_t pad;
};
struct CalendarAdditionalInfo {
    int32_t day_of_week;  // DayOfWeek: domingo 0
    int32_t year_day;     // 0..365
    char time_zone[8];
    bool is_dst;
    int32_t utc_offset_seconds;
};
struct LocationName {
    char value[36];
};
// En la consola son 0x4000 bytes opacos que el juego copia con memcpy (main+0x6b3dfb8): aquí solo se usa el
// principio, el nombre de la zona, sin punteros.
struct TimeZoneRule {
    LocationName location;
};
static_assert(sizeof(CalendarTime) == 8 && sizeof(CalendarAdditionalInfo) == 24 && sizeof(LocationName) == 36);

class StandardUserSystemClock {
public:
    static Result GetCurrentTime(PosixTime* out);
};
class StandardNetworkSystemClock {
public:
    static Result GetCurrentTime(PosixTime* out);
    static bool IsAccuracySufficient();
};
// Punto del reloj monotónico: segundos desde una referencia y el id (Uuid) de esa referencia; dos puntos solo se
// comparan si tienen el mismo id (en la consola cambia al reiniciar el reloj).
struct SteadyClockTimePoint {
    int64_t value;
    uint8_t source_id[16];
};
static_assert(sizeof(SteadyClockTimePoint) == 24);
class StandardSteadyClock {
public:
    static Result GetCurrentTimePoint(SteadyClockTimePoint* out);
};
Result GetSpanBetween(int64_t* out_seconds, const SteadyClockTimePoint& from, const SteadyClockTimePoint& to);

namespace {

std::mutex g_tz_m;

CalendarTime to_calendar(const tm& t) {
    return CalendarTime{static_cast<int16_t>(t.tm_year + 1900), static_cast<int8_t>(t.tm_mon + 1),
                        static_cast<int8_t>(t.tm_mday),         static_cast<int8_t>(t.tm_hour),
                        static_cast<int8_t>(t.tm_min),          static_cast<int8_t>(t.tm_sec), 0};
}

std::string device_location() {
    char zone[PROP_VALUE_MAX] = {};
    return __system_property_get("persist.sys.timezone", zone) > 0 ? zone : "UTC";
}

// Hora local en una zona. La del teléfono es la de localtime_r; otra (rara: el juego carga la del teléfono)
// se calcula cambiando TZ un momento.
void local_in(const char* zone, time_t t, tm* out) {
    if (!*zone || device_location() == zone) {
        localtime_r(&t, out);
        return;
    }
    std::lock_guard lock(g_tz_m);
    const char* old = getenv("TZ");
    std::string saved = old ? old : "";
    setenv("TZ", zone, 1);
    tzset();
    localtime_r(&t, out);
    if (old)
        setenv("TZ", saved.c_str(), 1);
    else
        unsetenv("TZ");
    tzset();
}

void fill(CalendarTime* out, CalendarAdditionalInfo* info, const tm& local) {
    *out = to_calendar(local);
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = local.tm_wday;
        info->year_day = local.tm_yday;
        strncpy(info->time_zone, local.tm_zone ? local.tm_zone : "UTC", sizeof info->time_zone - 1);
        info->is_dst = local.tm_isdst > 0;
        info->utc_offset_seconds = static_cast<int32_t>(local.tm_gmtoff);
    }
}

}  // namespace

Result Initialize() { return ResultSuccess; }
Result Finalize() { return ResultSuccess; }
bool IsInitialized() { return true; }

Result StandardUserSystemClock::GetCurrentTime(PosixTime* out) {
    out->value = ::time(nullptr);
    return ResultSuccess;
}
Result StandardNetworkSystemClock::GetCurrentTime(PosixTime* out) {
    out->value = ::time(nullptr);
    return ResultSuccess;
}
// El reloj de Android se sincroniza por red (NTP/operador): la hora de "red" es buena.
bool StandardNetworkSystemClock::IsAccuracySufficient() { return true; }

// Un solo id para todo el proceso: los puntos de esta sesión se pueden restar entre sí.
constexpr uint8_t kSteadySource[16] = {0x58, 0x43, 0x44, 0x45, 0x73, 0x74, 0x65, 0x61,
                                       0x64, 0x79, 0x2d, 0x63, 0x6c, 0x6f, 0x63, 0x6b};
Result StandardSteadyClock::GetCurrentTimePoint(SteadyClockTimePoint* out) {
    timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    out->value = ts.tv_sec;
    memcpy(out->source_id, kSteadySource, sizeof out->source_id);
    return ResultSuccess;
}
// Result de nn::time para puntos de relojes distintos (NotComparable, módulo 116).
Result GetSpanBetween(int64_t* out_seconds, const SteadyClockTimePoint& from, const SteadyClockTimePoint& to) {
    if (memcmp(from.source_id, to.source_id, sizeof from.source_id) != 0) return Result{116 | 200 << 9};
    *out_seconds = to.value - from.value;
    return ResultSuccess;
}

Result GetDeviceLocationName(LocationName* out) {
    memset(out, 0, sizeof *out);
    strncpy(out->value, device_location().c_str(), sizeof out->value - 1);
    return ResultSuccess;
}

Result LoadTimeZoneRule(TimeZoneRule* rule, const LocationName& name) {
    memset(&rule->location, 0, sizeof rule->location);
    memcpy(rule->location.value, name.value, sizeof name.value - 1);
    return ResultSuccess;
}

Result ToCalendarTime(CalendarTime* out, CalendarAdditionalInfo* info, const PosixTime& posix) {
    time_t t = static_cast<time_t>(posix.value);
    tm local{};
    localtime_r(&t, &local);
    fill(out, info, local);
    return ResultSuccess;
}

Result ToCalendarTime(CalendarTime* out, CalendarAdditionalInfo* info, const PosixTime& posix,
                      const TimeZoneRule& rule) {
    char zone[sizeof rule.location.value + 1] = {};
    memcpy(zone, rule.location.value, sizeof rule.location.value);
    tm local{};
    local_in(zone, static_cast<time_t>(posix.value), &local);
    fill(out, info, local);
    return ResultSuccess;
}

// Hora local del teléfono -> PosixTime. La consola da las 0-2 interpretaciones posibles (cambio de horario);
// mktime elige una.
Result ToPosixTime(int* count, PosixTime* out, int max, const CalendarTime& c) {
    tm t{};
    t.tm_year = c.year - 1900;
    t.tm_mon = c.month - 1;
    t.tm_mday = c.day;
    t.tm_hour = c.hour;
    t.tm_min = c.minute;
    t.tm_sec = c.second;
    t.tm_isdst = -1;
    *count = 0;
    if (max < 1) return ResultSuccess;
    out[0].value = static_cast<int64_t>(mktime(&t));
    *count = 1;
    return ResultSuccess;
}

CalendarTime ToCalendarTimeInUtc(const PosixTime& posix) {
    time_t t = static_cast<time_t>(posix.value);
    tm utc{};
    gmtime_r(&t, &utc);
    return to_calendar(utc);
}

PosixTime ToPosixTimeFromUtc(const CalendarTime& c) {
    tm t{};
    t.tm_year = c.year - 1900;
    t.tm_mon = c.month - 1;
    t.tm_mday = c.day;
    t.tm_hour = c.hour;
    t.tm_min = c.minute;
    t.tm_sec = c.second;
    return PosixTime{static_cast<int64_t>(timegm(&t))};
}

int GetDayOfWeek(int year, int month, int day) {
    tm t{};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    t.tm_hour = 12;
    time_t s = timegm(&t);
    gmtime_r(&s, &t);
    return t.tm_wday;
}

bool IsValidDate(int year, int month, int day) {
    if (month < 1 || month > 12 || day < 1) return false;
    static const int kDays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    return day <= kDays[month - 1] + (month == 2 && leap ? 1 : 0);
}

}  // namespace nn::time
