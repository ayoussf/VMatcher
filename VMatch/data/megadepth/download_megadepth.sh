#!/bin/bash

while true; do
    wget -c https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz --no-check-certificate
    if [ $? -eq 0 ]; then
        echo "Download successful!"
        break
    else
        echo "Download failed. Retrying in 5 seconds..."
        sleep 5
    fi
done