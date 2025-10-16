
cd ~/mitm_detection_project  

#Para capturar el trafico de red:
python3 scripts/capture_traffic.py --mode normal --duration 300 --interface wlan0


source /home/kevin/mitm_detection_project/.venv/bin/activate

source .venv/bin/activate 

pip install --upgrade pip

pip install tensorflow

pip install pandas numpy scikit-learn matplotlib seaborn jupyter


python scripts/train_model.py