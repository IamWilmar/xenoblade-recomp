// El cuadro de texto que hace de teclado de la consola. No va en la APK: tools/teclado_dex.sh lo compila a un dex que
// queda dentro de libnnsdk (teclado_dex.inc) y teclado_android.cpp lo carga en memoria con InMemoryDexClassLoader.
// Así una actualización que solo cambia las .so también lo trae.
//
// Todo pasa en el hilo de la interfaz. resultado() avisa al código nativo una sola vez por cuadro:
// 1 = aceptado (con el texto), 2 = cancelado o cerrado, 3 = no se pudo mostrar.
package port;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.text.InputFilter;
import android.text.InputType;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.TextView;

public final class Teclado {
    static native void resultado(int estado, String texto);

    private static AlertDialog abierto;

    public static void abrir(final Activity actividad, final String titulo, final String mensaje, final String guia,
                             final String inicial, final String ok, final int max, final boolean clave,
                             final boolean lineas) {
        actividad.runOnUiThread(new Runnable() {
            public void run() {
                try {
                    mostrar(actividad, titulo, mensaje, guia, inicial, ok, max, clave, lineas);
                } catch (Throwable t) {
                    t.printStackTrace();
                    resultado(3, null);
                }
            }
        });
    }

    // El juego retiró su teclado: se cierra y cuenta como cancelado.
    public static void cerrar(Activity actividad) {
        actividad.runOnUiThread(new Runnable() {
            public void run() {
                if (abierto != null) abierto.dismiss();
            }
        });
    }

    private static void mostrar(Activity actividad, String titulo, String mensaje, String guia, String inicial,
                                String ok, int max, boolean clave, boolean lineas) {
        final EditText campo = new EditText(actividad);
        int tipo = InputType.TYPE_CLASS_TEXT;
        if (clave) tipo |= InputType.TYPE_TEXT_VARIATION_PASSWORD;
        if (lineas) tipo |= InputType.TYPE_TEXT_FLAG_MULTI_LINE;
        campo.setInputType(tipo);
        // En horizontal el teclado no tapa todo con su propio editor: Gboard ignora NO_FULLSCREEN y abría su editor a
        // pantalla completa, que escondía el título y el motivo cuando el juego rechaza el texto (visto en NieR).
        campo.setImeOptions(EditorInfo.IME_ACTION_DONE | EditorInfo.IME_FLAG_NO_FULLSCREEN
                            | EditorInfo.IME_FLAG_NO_EXTRACT_UI);
        if (max > 0) campo.setFilters(new InputFilter[] {new InputFilter.LengthFilter(max)});
        campo.setText(inicial);
        campo.setSelection(campo.length());
        if (guia.length() > 0) campo.setHint(guia);

        AlertDialog.Builder builder = new AlertDialog.Builder(actividad, AlertDialog.THEME_DEVICE_DEFAULT_DARK);
        if (titulo.length() > 0) builder.setTitle(titulo);
        if (mensaje.length() > 0) builder.setMessage(mensaje);
        final AlertDialog dialogo = builder.create();
        final boolean[] hecho = {false};
        final Runnable aceptar = new Runnable() {
            public void run() {
                if (hecho[0]) return;
                hecho[0] = true;
                resultado(1, campo.getText().toString());
            }
        };
        final Runnable cancelar = new Runnable() {
            public void run() {
                if (hecho[0]) return;
                hecho[0] = true;
                resultado(2, null);
            }
        };
        dialogo.setView(campo, 48, 16, 48, 0);
        dialogo.setButton(DialogInterface.BUTTON_POSITIVE,
                          ok.length() > 0 ? ok : actividad.getString(android.R.string.ok),
                          new DialogInterface.OnClickListener() {
                              public void onClick(DialogInterface d, int which) {
                                  aceptar.run();
                              }
                          });
        dialogo.setButton(DialogInterface.BUTTON_NEGATIVE, actividad.getString(android.R.string.cancel),
                          new DialogInterface.OnClickListener() {
                              public void onClick(DialogInterface d, int which) {
                                  cancelar.run();
                              }
                          });
        dialogo.setCanceledOnTouchOutside(false);
        dialogo.setOnDismissListener(new DialogInterface.OnDismissListener() {
            public void onDismiss(DialogInterface d) {
                if (abierto == dialogo) abierto = null;
                cancelar.run();  // si no se aceptó antes: atrás, B, cerrar()…
            }
        });
        // La tecla ✓ / Intro del teclado acepta (los botones quedan debajo del teclado en horizontal).
        campo.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            public boolean onEditorAction(TextView v, int accion, KeyEvent evento) {
                final boolean intro = evento != null && evento.getKeyCode() == KeyEvent.KEYCODE_ENTER;
                if (accion != EditorInfo.IME_ACTION_DONE && !intro) return false;
                if (intro && evento.getAction() != KeyEvent.ACTION_DOWN) return true;
                aceptar.run();
                dialogo.dismiss();
                return true;
            }
        });

        // Arriba y sin encoger (con ADJUST_RESIZE el campo se quedaba en 0 de alto): el teclado tapa los botones, no el
        // campo, y ✓ acepta.
        Window ventana = dialogo.getWindow();
        ventana.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE
                                 | WindowManager.LayoutParams.SOFT_INPUT_ADJUST_PAN);
        ventana.setGravity(Gravity.TOP | Gravity.CENTER_HORIZONTAL);
        abierto = dialogo;
        dialogo.show();
        // El mismo modo inmersivo que el juego: que no salgan las barras de Android mientras se ve.
        ventana.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                                                     | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                                                     | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                                                     | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                                                     | View.SYSTEM_UI_FLAG_FULLSCREEN
                                                     | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        campo.requestFocus();
    }
}
