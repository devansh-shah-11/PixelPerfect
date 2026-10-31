#!/bin/bash
# Download + unpack a Project AirSim v1.0.1 prebuilt Linux environment into /scratch/$USER/airsim/envs/<Env>.
# Usage (on an HPC login/CPU terminal, no GPU needed):  bash download_envs.sh Neighborhood
#   Env = Neighborhood | CityEnviron | LandscapeMountains    (Blocks is already installed)
# Safe to re-run: wget -c resumes, finished unpacks are skipped.
set -euo pipefail
ENV_NAME=${1:?usage: download_envs.sh <Neighborhood|CityEnviron|LandscapeMountains>}
BASE=https://github.com/iamaisim/ProjectAirSim/releases/download/v1.0.1
DEST=/scratch/$USER/airsim/envs/$ENV_NAME
DL=/scratch/$USER/airsim/downloads
mkdir -p "$DEST" "$DL"
cd "$DL"
if [ -n "$(find "$DEST" -maxdepth 2 -name '*.sh' 2>/dev/null | head -1)" ]; then
    echo "already unpacked: $DEST"; find "$DEST" -maxdepth 2 -name '*.sh'; exit 0
fi
case "$ENV_NAME" in
    CityEnviron)
        wget -c "$BASE/CityEnviron-Linux-1.0.1.zip.001"
        wget -c "$BASE/CityEnviron-Linux-1.0.1.zip.002"
        # sizes from the release page: 1900 MB + 1338 MB; the parts are a plain split, so concatenate
        cat CityEnviron-Linux-1.0.1.zip.001 CityEnviron-Linux-1.0.1.zip.002 > CityEnviron-Linux-1.0.1.zip
        ZIP=CityEnviron-Linux-1.0.1.zip ;;
    Neighborhood|LandscapeMountains)
        ZIP=$ENV_NAME-Linux-1.0.1.zip
        wget -c "$BASE/$ZIP" ;;
    *) echo "unknown env $ENV_NAME"; exit 1 ;;
esac
unzip -tq "$ZIP" | tail -1            # integrity check before unpacking
unzip -q -o "$ZIP" -d "$DEST"
chmod +x "$DEST"/*.sh 2>/dev/null || chmod +x "$DEST"/*/*.sh
echo "launchers:"; find "$DEST" -maxdepth 2 -name '*.sh'
du -sh "$DEST"
echo "zip files kept in $DL (delete with: rm $DL/*.zip*)"
