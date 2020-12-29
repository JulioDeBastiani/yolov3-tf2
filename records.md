# Records

## About
Here we store the tables of the time that took for training an epoch of the yolo model in the gym repository. In each table we store:

- O.S. : The operating system.

- Validation batch size.

- Train dataset size.

- Val dataset size.

- AVR (Average Read Rate): Average time for the secundary memory to read 100 samples. (gnome-disk used to benchmark)

- Memory speed: used `dmidecode --type 17` to check it.

- GPU memory: The memory size of the GPU
​
- GPU frequency: The frequency of the GPU. Used the comand `nvidia-settings -q GPUCurrentClockFreqs`

- GPU: GPU name

- Time: used the average time between 2 finished epochs from the finishing of the last one till the finish of the current.

## PERSONDET

| O.S. | Batch size | Train dataset size | Val dataset size | AVR | Memory speed | GPU memory | GPU frequency | GPU | Time | 
| ------------ | ----------------------| ------------------ | ------ | ---------------- | ---- | ------ | ------ |---------------- | ---- |
| Ubuntu 18.04.5 LTS| 20 | 49943 | 10702 | 539,3 MB/s | 3000 MT/s | 6GB (5931MB) | 300,405 MHz | GeForce RTX 2060 | 21 min 22.5 sec |

## TINY-PERSONDET

| O.S. | Batch size | Train dataset size | Val dataset size | AVR | Memory speed | GPU memory | GPU frequency | GPU | Time | 
| ------------ | ----------------------| ------------------ | ------ | ---------------- | ---- | ------ | ------ |---------------- | ---- |
| Ubuntu 18.04.5 LTS| 20 | 49943 | 10702 | 539,3 MB/s | 3000 MT/s | 6GB (5931MB) | 300,405 MHz | GeForce RTX 2060 | 11 min 50.5 sec |

## FACEDET

No data available

## TINY-FACEDET

No data available
