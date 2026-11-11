package io.github.iamwilmar.xenobladeport;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

// Busca en una carpeta los archivos del juego que el usuario extrajo de su propia consola: el ExeFS y la RomFS del juego
// con la actualización aplicada y, si el juego los tiene, la RomFS de cada DLC; y comprueba que son de este juego y de la
// versión del port. No descifra nada ni abre NSP/XCI/NCA: solo lee carpetas con archivos ya extraídos. Lo que encuentra
// va a <carpeta>/rutas.txt, que lee el runtime (rutas.h). Sin nada de Android, para probarlo en un PC:
// java ... DumpScanner <carpeta>.
final class DumpScanner {
    private static final boolean ES = Locale.getDefault().getLanguage().equals("es");

    static String tr(String es, String en) {
        return ES ? es : en;
    }

    static final class Result {
        String exefs, romfs;
        final TreeMap<Integer, String> dlc = new TreeMap<>();  // índice del DLC -> su RomFS
        String exefsProblem, romfsProblem;  // por qué no vale lo que se encontró (null: no había nada)
        int unknownDlc;  // RomFS de DLC sin su id de título en el nombre de ninguna carpeta

        boolean ready() {
            return exefs != null && romfs != null;
        }
    }

    // Id de título de la Switch en un nombre de carpeta (las herramientas de volcado lo ponen: "... (0100…3001) (RomFS)").
    private static final Pattern TITLE_ID = Pattern.compile("(?i)(?<![0-9a-f])(01[0-9a-f]{14})(?![0-9a-f])");
    private static final int MAX_DEPTH = 8, MAX_DIRS = 50000;

    private final Result result = new Result();
    private int visited;

    static Result scan(File root) {
        DumpScanner scanner = new DumpScanner();
        scanner.visit(root, 0, -1);
        return scanner.result;
    }

    // ¿Está esta marca (un archivo, o una ruta relativa "carpeta/archivo") en la carpeta? names: lo que hay dentro.
    private static boolean has(File dir, Set<String> names, String marker) {
        if (marker == null) return false;
        int slash = marker.indexOf('/');
        if (slash < 0) return names.contains(marker);
        return names.contains(marker.substring(0, slash)) && new File(dir, marker).isFile();
    }

    // tid: el id de título del nombre de esta carpeta o de la más cercana de encima que tenga uno (-1: ninguna).
    private void visit(File dir, int depth, long tid) {
        if (depth > MAX_DEPTH || ++visited > MAX_DIRS) return;
        String[] names = dir.list();
        if (names == null) return;
        Matcher m = TITLE_ID.matcher(dir.getName());
        if (m.find()) tid = Long.parseLong(m.group(1), 16);
        Set<String> set = new HashSet<>(Arrays.asList(names));
        if (set.contains("main") && set.contains("main.npdm")) checkExefs(dir);
        if (has(dir, set, Game.ROMFS_MARKER)) {  // las RomFS no se recorren: miles de carpetas
            checkRomfs(dir, set);
            return;
        }
        if (has(dir, set, Game.DLC_MARKER)) {
            addDlc(dir, tid);
            return;
        }
        Arrays.sort(names);
        for (String name : names) {
            File sub = new File(dir, name);
            if (!name.startsWith(".") && sub.isDirectory()) visit(sub, depth + 1, tid);
        }
    }

    private void checkExefs(File dir) {
        if (result.exefs != null) return;
        byte[] npdm = read(new File(dir, "main.npdm"), 1 << 16), nso = read(new File(dir, "main"), 0x60);
        if (npdm == null || nso == null || npdm.length < 0x80 || nso.length < 0x60) return;
        if (!magic(npdm, 0, "META") || !magic(nso, 0, "NSO0")) {
            if (result.exefsProblem == null)
                result.exefsProblem = tr("El ExeFS de \"" + dir.getName() + "\" está cifrado o no es un ExeFS extraído.",
                        "The ExeFS in \"" + dir.getName() + "\" is encrypted or not an extracted ExeFS.");
            return;
        }
        int aci = (int) le(npdm, 0x70, 4);
        if (aci < 0 || aci + 0x18 > npdm.length || !magic(npdm, aci, "ACI0")) return;
        StringBuilder build = new StringBuilder();
        for (int k = 0x40; k < 0x54; k++) build.append(String.format("%02x", nso[k] & 0xff));
        if (le(npdm, aci + 0x10, 8) != Game.APP_ID) {
            if (result.exefsProblem == null)
                result.exefsProblem = tr("El ExeFS de \"" + dir.getName() + "\" es de otro juego.",
                        "The ExeFS in \"" + dir.getName() + "\" belongs to a different game.");
        } else if (!build.toString().equals(Game.BUILD_ID)) {
            result.exefsProblem = tr("El ExeFS es de otra versión del juego: este port necesita la " + Game.VERSION
                    + ". Vuelve a extraerlo con la actualización " + Game.VERSION + " instalada.",
                    "The ExeFS is from a different game version: this port needs " + Game.VERSION
                    + ". Extract it again with update " + Game.VERSION + " installed.");
        } else {
            result.exefs = dir.getPath();
            result.exefsProblem = null;
        }
    }

    private void checkRomfs(File dir, Set<String> names) {
        if (result.romfs != null) return;
        if (Game.UPDATE_MARKER == null || has(dir, names, Game.UPDATE_MARKER)) {
            result.romfs = dir.getPath();
            result.romfsProblem = null;
        } else {
            result.romfsProblem = tr("La RomFS no trae la actualización " + Game.VERSION
                    + ": vuelve a extraerla con la actualización aplicada.",
                    "The RomFS does not include update " + Game.VERSION + ": extract it again with the update applied.");
        }
    }

    // El índice del DLC es su id menos el del juego con 0x1000 sumado (aoc.cpp). Sin id en el nombre vale también una
    // carpeta aoc/<índice>.
    private void addDlc(File dir, long tid) {
        long index = tid >= 0 ? tid - Game.APP_ID - 0x1000 : -1;
        File parent = dir.getParentFile();
        if ((index <= 0 || index >= 4096) && parent != null && parent.getName().equals("aoc")) {
            try {
                index = Integer.parseInt(dir.getName());
            } catch (NumberFormatException e) {
                index = -1;
            }
        }
        if (index > 0 && index < 4096) {
            result.dlc.putIfAbsent((int) index, dir.getPath());
        } else {
            result.unknownDlc++;
        }
    }

    static void writePaths(File data, Result r) throws IOException {
        File tmp = new File(data, "rutas.txt.tmp");
        try (FileWriter w = new FileWriter(tmp)) {
            w.write("# Written by the app's start screen: where each part of the game files is.\n");
            w.write("exefs=" + r.exefs + "\n");
            w.write("romfs=" + r.romfs + "\n");
            for (Map.Entry<Integer, String> e : r.dlc.entrySet()) w.write("aoc " + e.getKey() + "=" + e.getValue() + "\n");
        }
        if (!tmp.renameTo(new File(data, "rutas.txt"))) throw new IOException("cannot write rutas.txt in " + data);
    }

    // Lo que apuntó la última vez, si sigue en su sitio (null: hay que volver a buscar).
    static Result load(File data) {
        Result r = new Result();
        try (BufferedReader in = new BufferedReader(new FileReader(new File(data, "rutas.txt")))) {
            for (String line; (line = in.readLine()) != null; ) {
                int eq = line.indexOf('=');
                if (line.startsWith("#") || eq < 0) continue;
                String key = line.substring(0, eq), value = line.substring(eq + 1);
                if (key.equals("exefs")) r.exefs = value;
                else if (key.equals("romfs")) r.romfs = value;
                else if (key.startsWith("aoc ")) r.dlc.put(Integer.parseInt(key.substring(4)), value);
            }
        } catch (IOException | NumberFormatException e) {
            return null;
        }
        String romfsCheck = Game.UPDATE_MARKER != null ? Game.UPDATE_MARKER : Game.ROMFS_MARKER;
        boolean present = r.ready() && new File(r.exefs, "main").isFile() && new File(r.romfs, romfsCheck).isFile();
        return present ? r : null;
    }

    private static byte[] read(File file, int max) {
        try (InputStream in = new FileInputStream(file)) {
            byte[] buf = new byte[max];
            int n = 0;
            for (int k; n < max && (k = in.read(buf, n, max - n)) > 0; ) n += k;
            return Arrays.copyOf(buf, n);
        } catch (IOException e) {
            return null;
        }
    }

    private static boolean magic(byte[] b, int at, String m) {
        for (int k = 0; k < 4; k++)
            if (b[at + k] != m.charAt(k)) return false;
        return true;
    }

    private static long le(byte[] b, int at, int size) {
        long v = 0;
        for (int k = size - 1; k >= 0; k--) v = (v << 8) | (b[at + k] & 0xff);
        return v;
    }

    public static void main(String[] args) {
        Result r = scan(new File(args[0]));
        System.out.println("exefs: " + r.exefs + (r.exefsProblem != null ? " — " + r.exefsProblem : ""));
        System.out.println("romfs: " + r.romfs + (r.romfsProblem != null ? " — " + r.romfsProblem : ""));
        System.out.println("dlc: " + r.dlc.size() + (r.dlc.isEmpty() ? "" : " (" + r.dlc.firstKey() + ".." + r.dlc.lastKey() + ")")
                + ", unknown: " + r.unknownDlc + ", ready: " + r.ready());
    }
}
