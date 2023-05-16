import os
import yaml

def config():
        # load config from config.yaml
    with open('src/config/config.yaml', 'r') as f:
        c = yaml.safe_load(f)
    # init with config from utils/config.yaml
    
    env_usr = os.environ.get("USER")
    c['aws']= True if "ubuntu" in env_usr else False
    c['data-directory']=c['data-directory-ec2'] if c['aws'] else c['data-directory-local']  
        
    # for every key in config that ends with 'dir', add the data directory to the beginning
    for key in c:
        if key.endswith('dir'):
            c[key] = os.path.join(c['data-directory'], c[key])
            os.makedirs(c[key], exist_ok=True) 
        
            
    os.makedirs(os.path.join(os.path.abspath(os.getcwd()), "model"), exist_ok=True)     

    return c