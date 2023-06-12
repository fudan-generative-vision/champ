# Downloading all tars from https://www.dropbox.com/sh/4y90ay6a6nki9v7/AADUGVlO401tx6a14SXoua-ga
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACJetGWdtee9TP_ejmBMdKOa/hmr2_training_data_part1.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AAA7vh5e4jubZ3cYRyOtYYQZa/hmr2_training_data_part2.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACtCMwexHgDCMxh5CYP1VEya/hmr2_training_data_part3.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AAB7C9Y33MEX5G9_UCHQ2KyLa/hmr2_training_data_part4a.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AABkok2N00PHFGHiKV8-RjpNa/hmr2_training_data_part4b.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AADYg_LKlGbjpjvrOQWoOgrna/hmr2_training_data_part4c.tar
wget https://www.dropbox.com/sh/4y90ay6a6nki9v7/AACARYZJMoxq9nS6cDMAL8Pya/hmr2_training_data_part4d.tar

for f in hmr2_training_data_part*.tar; do
    tar --warning=no-unknown-keyword --exclude=".*" -xvf $f
done
