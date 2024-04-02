# Google drive links to download the training data
gdown https://drive.google.com/uc?id=19e5w1x8JIJkYUT9TIPcFhQF1SXny_Qwz # hmr2_training_data_part1.tar
gdown https://drive.google.com/uc?id=10glv5LKLhqPHHpaWmvkg4wT3wSmL6E2f # hmr2_training_data_part2.tar
gdown https://drive.google.com/uc?id=19AJMbE0nKD6Lao22aAaPLngiuvrnxF5V # hmr2_training_data_part3.tar
gdown https://drive.google.com/uc?id=1ntyjT3-JbtDxoHqmCJzJ8JemMKxM78Xj # hmr2_training_data_part4a.tar
gdown https://drive.google.com/uc?id=1H1xBQqePHcayR65oRn3cnxdgau_WLrLD # hmr2_training_data_part4b.tar
gdown https://drive.google.com/uc?id=1TnYqqG-wQMfMjd0EHbMEqxDt8DXG3vbW # hmr2_training_data_part4c.tar
gdown https://drive.google.com/uc?id=1tY6N8RDtgWfHSnTLgUNfbGhPI5X9fMg9 # hmr2_training_data_part4d.tar

# Alternatively, consider using the dropbox links:
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACJetGWdtee9TP_ejmBMdKOa/hmr2_training_data_part1.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AAA7vh5e4jubZ3cYRyOtYYQZa/hmr2_training_data_part2.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACtCMwexHgDCMxh5CYP1VEya/hmr2_training_data_part3.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AAB7C9Y33MEX5G9_UCHQ2KyLa/hmr2_training_data_part4a.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AABkok2N00PHFGHiKV8-RjpNa/hmr2_training_data_part4b.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AADYg_LKlGbjpjvrOQWoOgrna/hmr2_training_data_part4c.tar
#wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACARYZJMoxq9nS6cDMAL8Pya/hmr2_training_data_part4d.tar

for f in hmr2_training_data_part*.tar; do
    tar --warning=no-unknown-keyword --exclude=".*" -xvf $f
done
