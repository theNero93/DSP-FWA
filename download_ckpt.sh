#!/bin/sh

WGET_URL="https://drive.google.com/u/0/uc?id=13wbA5kHRGODBDdiJ2gPeB1XK4KiCh-Im&export=download"

mkdir ./ckpt
wget $WGET_URL -O ./ckpt/SPP-res-50.pth
