// nn::settings: códigos de idioma. El juego compara el idioma que le da nn::oe::GetDesiredLanguage con los
// suyos (LanguageCode::Make de cada Language). En la consola la tabla la tiene el servicio de settings; es la
// fija de Nintendo, en el orden del enum Language.

#include <cstring>

#include "nnsdk/types.h"

namespace nn::settings {

enum Language : int {
    Language_Japanese,
    Language_AmericanEnglish,
    Language_French,
    Language_German,
    Language_Italian,
    Language_Spanish,
    Language_Chinese,
    Language_Korean,
    Language_Dutch,
    Language_Portuguese,
    Language_Russian,
    Language_Taiwanese,
    Language_BritishEnglish,
    Language_CanadianFrench,
    Language_LatinAmericanSpanish,
    Language_SimplifiedChinese,
    Language_TraditionalChinese,
    Language_BrazilianPortuguese,
    Language_Count,
};

struct LanguageCode {
    char string[8];
    static LanguageCode Make(Language language);
};

bool operator==(const LanguageCode& code, const Language& language);
bool operator==(const Language& language, const LanguageCode& code);

namespace {

const char* const kCodes[Language_Count] = {"ja", "en-US", "fr",      "de",     "it",      "es",
                                            "zh-CN", "ko", "nl",      "pt",     "ru",      "zh-TW",
                                            "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR"};

}  // namespace

LanguageCode LanguageCode::Make(Language language) {
    LanguageCode code{};
    if (language >= 0 && language < Language_Count) strncpy(code.string, kCodes[language], sizeof code.string);
    return code;
}

bool operator==(const LanguageCode& code, const Language& language) {
    return language >= 0 && language < Language_Count && strncmp(code.string, kCodes[language], 8) == 0;
}
bool operator==(const Language& language, const LanguageCode& code) { return code == language; }

// Todos los idiomas que sabe la consola, en el orden del enum (el juego elige entre ellos los que tiene).
int GetAvailableLanguageCodeCount() { return Language_Count; }
int GetAvailableLanguageCodes(LanguageCode* out, int count) {
    int n = count < Language_Count ? count : Language_Count;
    for (int k = 0; k < n; k++) out[k] = LanguageCode::Make(static_cast<Language>(k));
    return n < 0 ? 0 : n;
}

}  // namespace nn::settings
