from DSSATTools.soil import SoilProfile, SoilLayer
from .utils import find_soil_textural_class, calculate_sks, slu1


class DSSATSoil_fromSOILGRIDS(SoilProfile):
    @property
    def general_description(self):
        """ddsat soil descriptios abbs
        taken from https://github.com/daquinterop/Py_DSSATTools/blob/main/DSSATTools/soil.py

        Returns:
            _type_: _description_
        """
        return {
        'SLLL':  'Lower limit of plant extractable soil water, cm3 cm-3',  ## wv1500 soil grid 10-3 cm3cm-3 # spermanent wilting point *0.001
        'SDUL':  'Drained upper limit, cm3 cm-3', ### wv0033 # field capacity *0.001
        'SSAT':  'Upper limit, saturated, cm3 cm-3', ## soil saturatutaion wv0010 # Saturated water content wv0010 *0.001
        'SRGF':  'Root growth factor, soil only, 0.0 to 1.0',
        'SBDM':  'Bulk density, moist, g cm-3', ## nbulkdensity bdod (Bulk density of the fine earth fraction) cg/cm³
        'SLOC':  'Organic carbon, %', # soc Soil organic carbon content in the fine earth fraction	dg/kg 0.01 
        'SLCL':  'Clay (<0.002 mm), %', #Proportion of sand particles (> 0.05 mm) in the fine earth fraction	g/kg * 0.01
        'SLSI':  'Silt (0.05 to 0.002 mm), %', #Proportion of silt particles (≥ 0.002 mm and ≤ 0.05 mm) in the fine earth fraction	g/k
        'SLCF':  'Coarse fraction (>2 mm), %', #Volumetric fraction of coarse fragments (> 2 mm)	cm3/dm3 (vol‰) *0.1
        'SLNI':  'Total nitrogen, %', #Total nitrogen (N)	cg/kg * 0.01
        'SLHW': 'Phh2o', #Soil pH	pHx10 *0.1
        'SMHB':  'pH in buffer determination method, code', ### -99
        'SCEC':  'Cation exchange capacity, cmol kg-1', # Cation Exchange Capacity of the soil	mmol©/kg	*.10
        'SADC':  'Anion adsorption coefficient (reduced nitrate flow), cm3 (H2O) g [soil]-1',
        'SSKS':  'Sat. hydraulic conductivity, macropore, cm h-1'}

    def soilgrid_dict(self, dict_names  = None):
        if dict_names:
            self._dict_names = dict_names
        else:
            self._dict_names = {'clay': ['SLCL', 0.01],
         'silt': ['SLSI', 0.01],
         'nitrogen': ['SLNI', 0.01],
         'bdod': ['SBDM', 0.01],
         'cfvo': ['SLCF', 0.01],
         'phh2o': ['SLHW', 0.1],
         'soc': ['SLOC', 0.01],
         'cec': ['SCEC', 0.1],
         'wv1500': ['SLLL',0.001],
         'wv0033': ['SDUL',0.001],
         'wv0010': ['SSAT',0.001]}
        
        return self._dict_names

    @property
    def albedos(self):
        return (0.12, 0.12, 0.13, 0.13, 0.12, 0.13, 0.13, 0.14, 0.13, 0.13, 0.16, 0.19, 0.13)
    @property
    def runnoff_curves(self):
        return (73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 73.0, 68.0, 73.0, 68.0, 68.0, 73.0)
    
    @property
    def texture_classess(self):
        return ("clay", "silty clay", "sandy clay", "clay loam", "silty clay loam", "sandy clay loam", "loam", "silty loam", "sandy loam", "silt", "loamy sand", "sand", "unknown")
    
    @property
    def sum_text_classess(self):
        return ("C", "SIC", "SC", "CL", "SICL", "SCL", "L", "SIL", "SL", "SI", "LS", "S", "unknown")
    
    @staticmethod
    def get_init_from_texture(sand, clay):
        return find_soil_textural_class(sand, clay)

    def from_df_to_dssat_list(self, df, depth_col_name = 'depth'):
        colnames = df.columns
        soillayers = []
        for row in range(df.shape[0]):
            dict_dssat = {}
            for col in colnames:
                if col not in list(self._dict_names.keys()):
                    continue
                
                kdssat, fac = self._dict_names[col]
                val = fac*df[col].values[row]
                dict_dssat[kdssat]  = val

            dict_dssat['SSKS'] = calculate_sks(None, 
                            dict_dssat[self._dict_names['silt'][0]], 
                            clay = dict_dssat[self._dict_names['clay'][0]], 
                            bulk_density = dict_dssat[self._dict_names['bdod'][0]], 
                            field_capacity = dict_dssat.get('SDUL',None),
                            permanent_wp = dict_dssat.get('SLLL',None))

            depth = df[depth_col_name].values[row]
            soillayers.append((depth, dict_dssat))
        return soillayers
    
    def add_soil_layers_from_df(self, df, depth_col_name = 'depth'):
        #soil.SoilLayer(20, {'SLCL': 50, 'SLSI': 45}),
        listvals = self.from_df_to_dssat_list(df, depth_col_name= depth_col_name)

        soillayers = [SoilLayer(d, v) for d,v in listvals]
        for layer in soillayers: self.add_layer(layer)
            
    def soil_general_line_parameters(self):
        self._texture = self.get_init_from_texture(self._sand, self._clay)

        self._pos_text = self.texture_classess.index(self._texture)

        self.SALB = self.albedos[self._pos_text]

        self.SLU1 = slu1(self._clay, self._sand)

        #self.SLDR = 0.5
        self.SLRO = self.runnoff_curves[self._pos_text]
        self.SLNF = 1.
        self.SLPF =0.92
        self.SMHB,  self.SMPX,  self.SMKE = "IB001", "IB001", "IB001"

    def set_general_location_patamenters(self, **kwargs):
        self.description = kwargs.get('description',None)
        self.description = "ISRIC V2 {} {}".format(self.sum_text_classess[self._pos_text],
                                                                self._texture) if self.description is None else self.description
        
        self.country = kwargs.get('country', 'COL')
        self.site = kwargs.get('site', 'CAL')
        self.lat = kwargs.get('lat', -99)
        self.lon = kwargs.get('lon', -99)

        self.id = kwargs.get('id', None)
        self.id = f'{self.country}-{self.site}' if self.id is None else self.id


    def __init__(self,params, **kwargs) -> None:

        self._sand = params.get('sand',None)
        if self._sand: del params['sand']
        self._clay = params.get('clay',None)
        if self._clay: del params['clay']
        self._silt = params.get('silt',None)
        if self._silt: del params['silt']

        super().__init__(pars = params)
        self.soil_general_line_parameters()
        self.soilgrid_dict()
        self.set_general_location_patamenters(**kwargs)
        