package io.github.iamwilmar.xenobladeport;

// What identifies this game in the user's files: the rest of the start screen is the same in every port.
// Lo que distingue a este juego en los archivos del usuario: el resto de la pantalla de inicio es igual en cada port.
final class Game {
    static final String NAME = "Xenoblade Chronicles: Definitive Edition";
    static final long APP_ID = 0x0100FF500E34A000L;
    // The port only works with this exact version (its addresses and patches are those of this main): recognised by the
    // build ID of the ExeFS main (bytes 0x40-0x53 of the NSO header).
    static final String VERSION = "1.1.2";
    static final String BUILD_ID = "92c78bb3dcbbc3f7a3caf601d7b85f7a36c20907";
    static final String ROMFS_MARKER = "bdat/bdat_common.bdat";  // relative to the RomFS root
    static final String UPDATE_MARKER = null;  // only present with the update applied (null: not checked)
    static final String DLC_MARKER = null;  // in the root of each DLC RomFS (null: game without DLC)

    private Game() {}
}
