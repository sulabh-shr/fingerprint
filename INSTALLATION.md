# Installation Instructions
*If using hopper*  
`module load gnu10/10.3.0-ya`  
`module load python/3.10.1-qb`

Create new virtual environment using venv:   
`cd <path to environment/env-name>`  
`python -m venv finger`  
`source finger/bin/activate`

**OR**, create new virtual environment using conda:   
`conda create -n <env-name> python=3.10`  
`conda activate <env-name>`

Install dependencies:  
`python -m pip install --upgrade pip`  
`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`  
`python -m pip install timm tensorboard scikit-learn scikit-image opencv-contrib-python natsort easydict matplotlib`  

Install fingerprint project:  
Change directory to fingerprint project, the one containing *setup.py*: `cd <path-to-fingerprint>`    
`python -m pip install -e .`  
*Do not forget the `.` in the above command at the end.*