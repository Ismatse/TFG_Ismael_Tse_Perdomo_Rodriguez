import os
import shutil
import random


dataset_path = "E:\\UPM\\TFG_SOFTWARE\\Proyecto\\CholecSeg8k"

new_dataset_path = "E:\\UPM\\TFG_SOFTWARE\\Proyecto\\Dataset212_CholecSeg8k"

path_labels_test = "E:\\UPM\\TFG_SOFTWARE\\Proyecto"


os.makedirs(os.path.join(new_dataset_path, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(new_dataset_path, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(path_labels_test, "evaluation\\labelsTs"), exist_ok=True)


train_percentage = 0.8

train_cases = []
train_cases_path = []
label_cases = []
label_cases_path = []


for video_folder in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_folder)
    if not os.path.isdir(video_path):
        continue

    for subfolder in os.listdir(video_path):
        subfolder_path = os.path.join(video_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        images = [img for img in os.listdir(subfolder_path) if img.startswith("frame_") and img.endswith("_endo.png")]

        labels = [img for img in os.listdir(subfolder_path) if img.endswith("_color_mask.png")]

        train_cases.extend([img for img in images])
        label_cases.extend([label for label in labels])
        train_cases_path.extend([os.path.join(subfolder_path, img) for img in images])
        label_cases_path.extend([os.path.join(subfolder_path, label) for label in labels])

count = 0

for archivo in range(len(train_cases)):
    nuevo_nombre = f"CS_{str(count).zfill(4)}_0000.png"
    ruta_nueva = os.path.join(os.path.dirname(train_cases_path[archivo]), nuevo_nombre)
    os.rename(train_cases_path[archivo], ruta_nueva)
    train_cases_path[archivo] = ruta_nueva
    train_cases[archivo] = nuevo_nombre
    count +=1

contador = 0

for archivo in range(len(label_cases)):
    nuevo_nombre = f"CS_{str(contador).zfill(4)}.png"
    ruta_nueva = os.path.join(os.path.dirname(label_cases_path[archivo]), nuevo_nombre)
    os.rename(label_cases_path[archivo], ruta_nueva)
    label_cases_path[archivo] = ruta_nueva
    label_cases[archivo] = nuevo_nombre
    contador +=1



num_train_cases = int(len(train_cases) * train_percentage)

indices = list(range(len(train_cases)))
random.shuffle(indices)

train_cases = [train_cases[i] for i in indices]
train_cases_path = [train_cases_path[i] for i in indices]
label_cases = [label_cases[i] for i in indices]
label_cases_path = [label_cases_path[i] for i in indices]

train_cases_tr = train_cases[:num_train_cases]
train_cases_path_tr = train_cases_path[:num_train_cases]
label_cases_tr = label_cases[:num_train_cases]
label_cases_path_tr = label_cases_path[:num_train_cases]

test_cases_ts = train_cases[num_train_cases:]
test_cases_path_ts = train_cases_path[num_train_cases:]
label_cases_ts = label_cases[num_train_cases:]
label_cases_path_ts = label_cases_path[num_train_cases:]


for i in range(len(train_cases_tr)):
    img_src = train_cases_path_tr[i]
    img_dest = os.path.join(new_dataset_path, "imagesTr", train_cases_tr[i])
    try:
        shutil.copy(img_src, img_dest)
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
    except PermissionError:
        print("Permission denied.")
    except Exception:
        print("Error occurred while copying file.")


for i in range(len(label_cases_tr)):
    label_src = label_cases_path_tr[i]
    label_dest = os.path.join(new_dataset_path, "labelsTr", label_cases_tr[i])
    shutil.copy(label_src, label_dest)
        

for i in range(len(test_cases_ts)):
    img_src = test_cases_path_ts[i]
    img_dest = os.path.join(new_dataset_path, "imagesTs", test_cases_ts[i])
    shutil.copy(img_src, img_dest)

for i in range(len(label_cases_ts)):
    img_src = label_cases_path_ts[i]
    img_dest = os.path.join(path_labels_test, "evaluation\\labelsTs", label_cases_ts[i])
    shutil.copy(img_src, img_dest)