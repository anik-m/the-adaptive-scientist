#!/bin/bash

echo "Starting a clean and targeted installation of OpenFOAM..."

# --- STEP 1: Clean up any broken repository configurations ---
echo "Cleaning up old repository files..."
sudo rm -f /etc/apt/sources.list.d/openfoam*
sudo rm -f /usr/share/keyrings/openfoam*

# --- STEP 2: Configure the official openfoam.org repository ---
echo "Configuring the official OpenFOAM repository..."
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | gpg --dearmor > /usr/share/keyrings/openfoam-archive-keyring.gpg"
sudo sh -c "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/openfoam-archive-keyring.gpg] https://dl.openfoam.org/ubuntu jammy main' > /etc/apt/sources.list.d/openfoam.list"
sudo apt-get -y update

# --- STEP 3: Hybrid Install to bypass the https->http redirect error ---
echo "Downloading the main OpenFOAM package directly from the source..."
wget -O openfoam12.deb "http://downloads.sourceforge.net/project/foam/foam/ubuntu/dists/jammy/main/binary-amd64/openfoam12_20250206_amd64.deb"

echo "Installing the package and its dependencies..."
sudo dpkg -i openfoam12.deb
# The '-f' flag tells apt to "fix broken dependencies"
sudo apt-get -y install -f

echo "OpenFOAM installation complete."

# --- STEP 4: Install Python dependencies ---
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup finished successfully!"
