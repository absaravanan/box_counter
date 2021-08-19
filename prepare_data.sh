cd box/obj_train_data/images
for i in {1..1000}; do cp box_counting.png "box_counting$i.png"; done
cd ./../labels
for i in {1..1000}; do cp box_counting.txt "box_counting$i.txt"; done
cd ./../../obj_valid_data/images
for i in {1..100}; do cp box_counting.png "box_counting$i.png"; done
cd ./../labels
for i in {1..100}; do cp box_counting.txt "box_counting$i.txt"; done
