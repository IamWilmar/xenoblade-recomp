package io.github.iamwilmar.xenobladeport;

import android.app.Activity;
import android.app.NativeActivity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.storage.StorageManager;
import android.os.storage.StorageVolume;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

// Pantalla de inicio. La primera vez pide el permiso de archivos y la carpeta donde el usuario copió los archivos de
// su juego (ExeFS y RomFS ya extraídos de su propia consola); DumpScanner dice qué hay y qué falta, y si está todo se
// apunta en files/carpeta.txt. Las siguientes, directa en "Iniciar juego". El juego es la NativeActivity (libnier.so):
// usa esa carpeta como <datos> (partidas, cachés, log). La app no descifra nada ni acepta NSP/XCI.
// Con el mando: la cruceta mueve el foco y A pulsa (Android traduce A a DPAD_CENTER). Textos en español o inglés según
// el idioma del sistema.
public final class SetupActivity extends Activity {
    private static final boolean ES = Locale.getDefault().getLanguage().equals("es");

    static String tr(String es, String en) {
        return ES ? es : en;
    }

    private LinearLayout root;
    private File browsing;  // carpeta del explorador; null: la lista de almacenamientos
    private boolean shown;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(24);
        root.setPadding(pad, pad, pad, pad);
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!Environment.isExternalStorageManager()) {
            showPermission();
            shown = false;
            return;
        }
        if (shown) return;  // volver de segundo plano no cambia de pantalla
        shown = true;
        File folder = savedFolder();
        DumpScanner.Result saved = folder != null ? DumpScanner.load(folder) : null;
        if (saved != null) {
            // Con la carpeta ya elegida y sus archivos en su sitio, directo al juego, sin pasar por esta pantalla. Vuelve a
            // salir si falta algo (más abajo), si se pidió desde el panel de ajustes del juego ("Cambiar carpeta" deja el
            // fichero elegir_carpeta en la carpeta) o si la app se abre dos veces seguidas en 10 segundos, que es la
            // salida si el juego no llega a arrancar.
            File ask = new File(folder, "elegir_carpeta");
            File stamp = new File(getFilesDir(), "ultimo_arranque.txt");
            long last = 0;
            try {
                last = Long.parseLong(new String(Files.readAllBytes(stamp.toPath())).trim());
            } catch (IOException | NumberFormatException e) {
                // sin arranque anterior
            }
            long now = System.currentTimeMillis();
            boolean again = now >= last && now - last < 10000;
            if (saved.ready() && !ask.exists() && !again) {
                try {
                    Files.write(stamp.toPath(), Long.toString(now).getBytes());
                } catch (IOException e) {
                    // sin la marca solo se pierde la salida de "dos veces seguidas"
                }
                startGame();
                return;
            }
            ask.delete();
            stamp.delete();
            showResult(folder, saved);
        } else if (folder != null && folder.isDirectory()) {
            scan(folder);  // movió algo desde la última vez
        } else {
            showBrowser(null);
        }
    }

    @Override
    @SuppressWarnings("deprecation")  // sigue llegando con targetSdk 33; el mando manda BACK con B
    public void onBackPressed() {
        if (browsing != null) showBrowser(parentOf(browsing));
        else super.onBackPressed();
    }

    private void showPermission() {
        clear(Game.NAME);
        text(tr("Para leer tu juego, la app necesita permiso para ver los archivos. Actívalo en la siguiente pantalla y vuelve.",
                "To read your game files the app needs the \"All files access\" permission. Turn it on in the next screen and come back."));
        Button grant = button(tr("Dar permiso", "Grant permission"));
        grant.setOnClickListener(v -> startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:" + getPackageName()))));
        grant.requestFocus();
    }

    private void showBrowser(File dir) {
        browsing = dir;
        clear(tr("Elige la carpeta de tu juego", "Choose your game folder"));
        text(tr("La carpeta donde copiaste el ExeFS y la RomFS de tu copia del juego (versión " + Game.VERSION
                        + "), ya extraídos de tu consola" + (Game.DLC_MARKER != null ? ", y los DLC si los tienes" : "")
                        + ". Ahí se guardarán también tus partidas.",
                "The folder where you copied the ExeFS and RomFS of your own copy of the game (version " + Game.VERSION
                        + "), already extracted on your console" + (Game.DLC_MARKER != null ? ", plus any DLC you own" : "")
                        + ". Your saves will be stored there too."));
        final List<File> targets = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        if (dir == null) {
            for (StorageVolume volume : getSystemService(StorageManager.class).getStorageVolumes()) {
                File d = volume.getDirectory();
                if (d == null) continue;
                targets.add(d);
                labels.add("💾  " + volume.getDescription(this));
            }
        } else {
            text(dir.getPath());
            Button use = button(tr("Usar esta carpeta", "Use this folder"));
            use.setOnClickListener(v -> scan(dir));
            targets.add(null);
            labels.add(tr("⬆  Atrás", "⬆  Back"));
            File[] subs = dir.listFiles(File::isDirectory);
            if (subs != null) {
                Arrays.sort(subs, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
                for (File sub : subs) {
                    if (sub.getName().startsWith(".")) continue;
                    targets.add(sub);
                    labels.add("📁  " + sub.getName());
                }
            }
        }
        ListView list = new ListView(this);
        list.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, labels));
        list.setOnItemClickListener((parent, view, position, id) -> {
            File target = targets.get(position);
            showBrowser(target == null ? parentOf(dir) : target);
        });
        root.addView(list, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        list.requestFocus();
    }

    private void scan(File dir) {
        browsing = null;
        clear(tr("Buscando tu juego…", "Looking for your game…"));
        text(dir.getPath());
        root.addView(new ProgressBar(this));
        new Thread(() -> {
            DumpScanner.Result r = DumpScanner.scan(dir);
            String error = null;
            if (r.ready()) {
                try {
                    DumpScanner.writePaths(dir, r);
                    saveFolder(dir);
                } catch (IOException e) {
                    error = tr("No puedo escribir en ", "Cannot write to ") + dir + ": " + e.getMessage();
                }
            }
            String failure = error;
            runOnUiThread(() -> {
                showResult(dir, r);
                if (failure != null) text("✗  " + failure);
            });
        }, "BuscarJuego").start();
    }

    private void showResult(File dir, DumpScanner.Result r) {
        browsing = null;
        clear(Game.NAME);
        text(dir.getPath());
        if (r.exefs != null) text(tr("✓  Juego, versión ", "✓  Game, version ") + Game.VERSION);
        else text("✗  " + (r.exefsProblem != null ? r.exefsProblem
                : tr("No encuentro el ExeFS del juego (la carpeta con main y main.npdm).",
                     "Game ExeFS not found (the folder that contains main and main.npdm).")));
        if (r.romfs != null) text(tr("✓  RomFS del juego", "✓  Game RomFS"));
        else text("✗  " + (r.romfsProblem != null ? r.romfsProblem
                : tr("No encuentro la RomFS del juego.", "Game RomFS not found.")));
        if (Game.DLC_MARKER != null) {
            text(r.dlc.isEmpty() ? tr("·  Sin DLC (son opcionales)", "·  No DLC (optional)") : "✓  " + r.dlc.size() + " DLC");
            if (r.unknownDlc > 0)
                text("·  " + r.unknownDlc + tr(" DLC sin su id de título en el nombre de la carpeta: no los cargo. "
                        + "Deja las carpetas de los DLC con el nombre que les puso la herramienta con la que los extrajiste.",
                        " DLC without their title ID in the folder name were skipped. "
                        + "Keep the DLC folders named as your extraction tool created them."));
        }
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(row);
        Button change = new Button(this);
        change.setText(r.ready() ? tr("Cambiar carpeta", "Change folder") : tr("Elegir otra carpeta", "Choose another folder"));
        change.setOnClickListener(v -> showBrowser(dir));
        if (r.ready()) {
            Button start = new Button(this);
            start.setText(tr("Iniciar juego", "Start game"));
            start.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);
            start.setOnClickListener(v -> startGame());
            row.addView(start);
            start.requestFocus();
        } else {
            change.requestFocus();
        }
        row.addView(change);
    }

    private void startGame() {
        startActivity(new Intent(this, NativeActivity.class));
        finish();
    }

    private File savedFolder() {
        try {
            String path = new String(Files.readAllBytes(new File(getFilesDir(), "carpeta.txt").toPath())).trim();
            return path.isEmpty() ? null : new File(path);
        } catch (IOException e) {
            return null;
        }
    }

    private void saveFolder(File dir) throws IOException {
        try (FileWriter w = new FileWriter(new File(getFilesDir(), "carpeta.txt"))) {
            w.write(dir.getPath() + "\n");
        }
    }

    // Subir una carpeta; desde la raíz de un almacenamiento se vuelve a la lista de almacenamientos.
    private File parentOf(File dir) {
        for (StorageVolume volume : getSystemService(StorageManager.class).getStorageVolumes())
            if (dir.equals(volume.getDirectory())) return null;
        return dir.getParentFile();
    }

    private void clear(String title) {
        root.removeAllViews();
        TextView t = new TextView(this);
        t.setText(title);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, 28);
        root.addView(t);
    }

    private void text(String s) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, 18);
        t.setPadding(0, dp(8), 0, 0);
        root.addView(t);
    }

    private Button button(String label) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(12);
        root.addView(b, lp);
        return b;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }
}
