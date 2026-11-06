EarthNG Map Builder — Windows

Want to play the included demo?
Put earthng_planet.zip and the demo country's ZIP in your BeamNG user folder's
mods directory. Keep both ZIPs packed. Open Freeroam > EarthNG - Planet.
Map Builder and Python are only needed to make or download other map packs.

Ready maps — no generation or Python needed
Ukraine (2.96 GB):
https://drive.google.com/file/d/1WriRBq6pBBjSUyySV81RMgzRViK83NpJ/view
All ready country maps:
https://drive.google.com/drive/folders/1n3A-msXrJtuQ2qJKn7YqzCooLZyliKMy?usp=drive_link
Download the individual country's ZIP, keep it packed and place it in mods
beside earthng_planet.zip. Always get the planet ZIP from GitHub Releases.
If Drive creates a wrapper ZIP when downloading several files, extract that
wrapper first; the earthng_<country>.zip files inside must remain packed.
Map Builder also has a Download ready maps (Google Drive) button.

Build your own map
1. Install Python 3.13 from https://www.python.org/downloads/windows/
   Include Tcl/Tk and the Python launcher in the installer.
2. Extract this entire archive to a writable folder. Do not run it inside ZIP.
3. Double-click MapBuilder.bat. First launch runs Setup.bat to install packages.
4. Select your BeamNG user folder (the folder containing mods).
5. Choose a country/region or select an area on the map, then Build my own.
   Wait for the installed message. The result is also in out/country-setup.
6. Restart BeamNG after adding or replacing a map ZIP.

Install earthng_planet.zip first: it supplies the offline map, city labels,
preview materials and the game's runtime. Internet is required for initial
setup and new source data. A whole country can take hours: Ukraine took
1.5 hours on a 16-core PC with 32 GB. Before it starts, a build checks the disk
(about 24 GB per GB of source data) and refuses rather than fill it.
Stop cancels a build before installation. Build again and it continues where
it stopped: finished blocks, regions and parts of the pack are kept.

If the window does not open, MapBuilder.bat keeps its console open with the
reason (missing Tcl/Tk, packages, files) and saves it to
out\mapbuilder\startup.log. Setup.bat repairs .venv and reinstalls packages.
Attach startup.log when reporting a problem.

Uninstall: delete this extracted folder. Generated packs already installed in
BeamNG remain in mods; remove their ZIPs there if no longer wanted.
This experimental release was tested with BeamNG.drive 0.39.4 and 0.40 on Windows.
OSM coverage and performance vary. No multiplayer compatibility is claimed.

Code: MIT (LICENSE). Data and assets: THIRD_PARTY_NOTICES.md.
Release packaging assisted by OpenAI Codex and Claude Code.
Source and issues: https://github.com/Alezx311/earthng
