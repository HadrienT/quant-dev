#!/bin/bash

# Variables
ZIP_NAME="dataColCF.zip"
DEST_PATH="../Cloud/scripts/$ZIP_NAME"

# Vérifie si les fichiers main.py, requirements.txt, et tickers.csv existent dans le répertoire courant
if [[ ! -f "main.py" || ! -f "requirements.txt" || ! -f "tickers.csv" ]]; then
  echo "Error: main.py, requirements.txt, or tickers.csv not found in the current directory."
  exit 1
fi

# Crée le zip en remplaçant l'ancien fichier
zip -j "$ZIP_NAME" main.py requirements.txt tickers.csv
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to create the zip archive."
  exit 1
fi

# Déplace le zip dans le dossier cible en remplaçant l'existant
mv -f "$ZIP_NAME" "$DEST_PATH"
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to move the zip archive to $DEST_PATH."
  exit 1
fi

echo "Zip archive created and moved successfully to $DEST_PATH."
