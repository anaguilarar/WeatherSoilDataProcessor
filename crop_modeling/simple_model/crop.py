import pandas as pd
SPECIES  = [
    'wheat',
    'rice',
    'maize',
    'soybean',
    'drybean',
    'peanut',
    'potato',
    'cassava',
    'tomato',
    'sweetcorn',
    'grbean',
    'carrot',
    'cotton',
    'banana',
    'dummy'
]


# Input for running crop should be a vector (temperature, radiation, CO2, ... etc at each day)
class Crop(object):
    
    def __init__(self, crop, cultivar = None, crop_params_path = None, cultivar_path = None):
        self.crop_path = crop_params_path or 'crop_modeling/simple_model/parameters/Species.csv'
        self.cultivar_path = cultivar_path or 'crop_modeling/simple_model/parameters/Cultivar.csv'
        self.crop = crop
        self.cultivar = cultivar        

        assert self.crop.lower() in SPECIES, "this model works using the crop reported in "
        self._init_crop_params()
        # Crop parameters
    
    def _init_crop_params(self):
        species_params = pd.read_csv(self.crop_path)
        cultivar_params = pd.read_csv(self.cultivar_path)
        
        self.crop_params = species_params.loc[species_params['Species'] == self.crop]
        
        self.cultivar_params = cultivar_params.loc[cultivar_params['Species'] == self.crop]
        if self.cultivar is None:
            self.cultivar = self.cultivar_params['Cultivar'].values[0] if self.cultivar_params['Cultivar'].shape[0]>1 else self.cultivar_params['Cultivar']
        
        self.cultivar_params = self.cultivar_params.loc[self.cultivar_params['Cultivar'] == self.cultivar] 
    
    def set_params(self):
        self.T_sum        = self.cultivar_params['Tsum'].to_numpy()[0]
        self.HI           = self.cultivar_params['HI'].to_numpy()[0]
        self.I_50A        = self.cultivar_params['I50A'].to_numpy()[0]
        self.I_50B        = self.cultivar_params['I50B'].to_numpy()[0]
        self.T_base       = self.crop_params['Tbase'].to_numpy()[0]
        self.T_opt        = self.crop_params['Topt'].to_numpy()[0]
        self.RUE          = self.crop_params['RUE'].to_numpy()[0]
        self.I_50maxH     = self.crop_params['I50maxH'].to_numpy()[0]
        self.I_50maxW     = self.crop_params['I50maxW'].to_numpy()[0]
        self.T_heat       = self.crop_params['MaxT'].to_numpy()[0]
        self.T_ext        = self.crop_params['ExtremeT'].to_numpy()[0]
        self.S_CO2        = self.crop_params['CO2_RUE'].to_numpy()[0]
        self.S_water      = self.crop_params['S_Water'].to_numpy()[0]
        if self.crop == 'banana':
            self.senescence_day = 397 ## thesis Stevens  2021 
        else:
            self.senescence_day = None
            
