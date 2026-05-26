
from .management import nitrogen_release


class BANANAN_Dictionary:
    """
    Dictionary of parameter definitions and units for banana N model.
    """
    params = {
        'sddpss': "Thermal time interval between planting/ emergence and sucker emergence in mat 'm' for banana plant in cycle 'c'",
        'dNRESBAN': {"definition": "Mineral N from banana residue mineralization", "units": "g"}
    }


class PlantParameters:
    """
    Contains light interception, growth, decomposition, and physiological parameters for the banana plant.

    Attributes
    ----------
    Ea : float
        Maximun light interception efficiency of banana canopy.
    Ec : float
        Proportion of PAR intercepted.
    kBAN : float
        Extinction coefficient of banana canopy (Turner, 1990).
    Pintmax : float
        Maximum proportion of light intercepted by the canopy at flowering (Turner, 1990).
    sdd_iff : float
        Thermal time interval between floral induction and flowering (°C/day).
    sdd_fh : float
        Thermal time interval between flowering and harvest.
    sdd_pif : float
        Thermal time interval between planting and flowering induction.
    andd : float
        Parameters of finger number as a function of dry biomass at floral induction.
    bndd : float
        Parameters of finger number as a function of dry biomass at floral induction.
    Bunchflo : float
        Bunch dry weight at flowering (g).
    RGR : float
        Relative fruit growth rate (g -1 °C d-1).
    DMfruitmax : float
        Maximal finger dry biomass (g).
    slban : float
        Specific leaf area at flowering (m2/g).
    laiban1 : float
        Initial leaf area index of banana (m2 leaf area per m2 ground area).
    laiban_max : int
        Leaf area index of banana for maximal photosynthetically active radiation intercepted.
    phampe : float
        Proportion of stem within the bunch.
    pc_ban : float
        Percentage of carbon in the banana tree.
    r_ban : float
        Decomposition rate constant of residue banana.
    residue_c_yield : float
        Assimilation yield of residue-C by microbial biomass (Y).
    bm_decomr : float
        Decomposition rate constant of microbial biomass (L).
    cn_r_ban : float
        C/N Ratio of banana residues.
    cn_r_hum : float
        C/N humus.
    cn_r_mbban : float
        C:N ratio of zymogenous microbial biomass (CNBBAN).
    h_ban : float
        Humification rate of microbial biomass.
    wr_ban : float
        N:C ratio for equations simplification.
    wb_ban : float
        N:C ratio for microbial biomass equations simplification.
    ZrBAN1 : float
        Partitioning of the banana root exploration of the upper layer.
    ZrBAN2 : float
        Partitioning of the banana root exploration of the lower layer.
    Ksom1 : float
        Mineralization rate of soil organic nitrogen.
    """
    Ea: float = 0.95  # Maximun light interception efficiency of banana canopy
    Ec: float = 0.48 # Proportion of PAR intercepted
    kBAN: float = 0.7 # Extinction coefficient of banana canopy # Turner (1990)
    Pintmax: float = 0.7 # Maximum proportion of light intercepted by the canopy at flowering (Turner, 1990)
    sdd_iff: float = 880.0 # Thermal time interval between floral induction and flowering -> Measured with data from Rapetti. (2022) °C/day 
    sdd_fh: float = 750.0 # Thermal time interval between flowering and harvest -> Dorel et al. (2016)
    sdd_pif: float = 1451.0 # Thermal time interval between planting and flowering induction -> Measured with BS data
    
    andd: float = 0.0136 ## Parameters of finger number as a function of dry biomass at floral induction
    bndd: float = 151.51 ##Parameters of finger number as a function of dry biomass at floral induction
    
    Bunchflo: float = 644 # Bunch dry weight at flowering
    RGR: float = 0.321 # relative fruit growth rate # g -1 °C d-1
    
    DMfruitmax: float = 35 # Maximal finger dry biomass # g

    psk: float = 0.2        # Percentage allocation of total biomass to sucker
    allocroot: float = 0.028 # % of assimilates allocated to roots 
    
    slban: float = 0.018 # Specific leaf area at flowering m2/g
    laiban1: float = 0.1 # Initial leaf area index of banana m2 leaf area per m2 ground area
    laiban_max: int = 7 # Leaf area index of banana for maximal photosynthetically active radiation intercepted m2 leaf area per m2 ground area. Measured with data from Ruillé et al. (2023)
    
    phampe: float = 0.06 #Proportion of stem within the bunch
    pc_ban: float = 0.42 #Percentage of carbon in the banana tree
    
    r_ban: float = 0.38 # Decomposition rate constant of residue banana
    
    residue_c_yield: float = 0.62 #Assimilation yield of residue-C by microbial biomas Y
    bm_decomr: float = 0.0076# Decomposition rate constant of microbial biomass L
    
    cn_r_ban: float = 18.3 #C/N Ratio of banana residues (Experiment B)
    cn_r_hum: float = 1/10 # : C/N humus (wh - > r code)
    cn_r_mbban: float = 7.8 if cn_r_ban < 14.8 else 30.1 - (275/cn_r_ban) # C:N ratio of zymogenous microbial biomass  (CNBBAN)
    
    h_ban: float = 1 - ((0.91 * cn_r_ban) / 16.2 + cn_r_ban) #Humification rate of microbial biomass
    
    wr_ban: float = 1/cn_r_ban #N:C ratio for equations simplification
    wb_ban: float = 1/cn_r_mbban
    
    ZrBAN1: float = 0.5 # Partitioning of the banana root exploration of the upper layer
    ZrBAN2: float = 0.5 # Partitioning of the banana root exploration of the lower layer

    Ksom1: float = 0.0002 # Mineralization rate of soil organic nitrogen 

class BananaCycle(PlantParameters):
    """
    Represents a single plant generation (mother, sucker).

    Attributes
    ----------
    cycle : int
        The generation index of the plant.
    sdd : float
        Thermal time accumulated.
    sdd_pss : float
        Thermal time interval between planting/emergence and sucker emergence.
    laiban : float
        Leaf area index.
    ban_biomass : float
        Total above-ground dry biomass.
    veg_biomass : float
        Aboveground vegetative dry biomass.
    bun_biomass : float
        Dry biomass of banana bunch.
    rac_biomass : float
        Root biomass.
    stress : float
        Nitrogen stress coefficient.
    sominiflo : int
        Indicator of floral induction (1 if true, 0 otherwise).
    recolte : int
        Indicator of harvest (1 if true, 0 otherwise).
    reject_triggered : bool
        Prevents spawning multiple suckers.
    reject : int
        Indicator whether the plant is rejected.
    CrBAN : float
        Carbon in the residue pool.
    dNRESBAN : float
        Mineral N from banana residue mineralization.
    """
    def __init__(self, cycle_id: int, sdd_pss: float):
        self.cycle = cycle_id
        self.sdd = 0
        self.sdd_pss = sdd_pss # Thermal time interval between planting/ emergence and sucker emergence in mat ‘m’ for banana plant in cycle ‘c’
        self.laiban = 0.1 if self.cycle == 1 else 0
        self.ban_biomass = 10.0 if self.cycle == 1 else 0.0 # Total above-ground dry biomass of banana
        self.veg_biomass = 10.0 if self.cycle == 1 else 0.0 # Aboveground vegetative dry biomass in mat
        self.bun_biomass = 0.0 # Dry biomass of banana bunch in mat 'm'
        self.rac_biomass = 0.0 # Root biomass
        
        self.stress = 1.0 # N stress coefficient
        self.sominiflo = 0 # Indicator of floral induction (1 if floral induction has occurred, 0 otherwise)
        self.sdd_post_iniflo = 0 # Thermal time since flowering
        self.recolte = 0 # Indicator of harvest (1 if harvest has occurred, 0 otherwise)
        self.som_recolte = 0
        
        self.reject_triggered = False # prevents spawning multiple suckers
        self.reject = 0 # Indicator of whether the plant is rejected (1) or not (0)
        self.ndd = 0 # Number of fingers
        self.dmfruit = 0 # Dry biomass of fruit in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        
        self.dDMBANtot = 0 # Total newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        self.dDMBAN = 0 # Net newly formed dry biomass in mat ‘m’ for banana plant in cycle ‘c’ at time step ‘t’
        self.alloc_bun = 0 # Dry biomass allocated to the banana bunch in mat ‘m’ for banana plant
        self.received_biomass = 0 # Dry biomass received from the parent plant # allocfromPM
        self.alloc_suc = 0.0 
        # Residues
        self.Cr0BAN = 0 # Initial carbon in the residue pool at harvest
        self.CrBAN = 0 # Carbon in the residue pool
        self.CbBAN = 0 # Carbon in the microbial biomass pool
        self.ChBAN = 0 # Carbon in the humus pool
        self.dNhumBAN = 0 # N change in humified soil organic matter due to banana residues
        self.dNrBAN = 0 # N change in banana residues
        self.dNbBAN = 0 # N change in soil microbial biomass due to banana residues
        self.dNRESBAN = 0 # Mineral N from banana residue mineralization
        
    def update_phenology(self, temperature: float) -> None: 
        stress_factor_value = self.stress if self.sominiflo < 1 else 1.0
        stress_factor_value = max(0.1, stress_factor_value) 
        if self.sdd < 0: stress_factor_value = 1.0 
        
        self.sdd += max(0, temperature) * stress_factor_value 
        
        self.reject = 1 if self.sdd >= self.sdd_pss else 0
        
        self.sominiflo = self.sominiflo + 1 if self.sdd >= self.sdd_pif else 0 
        self.sdd_post_iniflo = self.sdd_post_iniflo + temperature if self.sominiflo >= 1 else 0 
        self.somfloraison = self.somfloraison + 1 if self.sdd_post_iniflo >= self.sdd_iff else 0 
        self.sdd_post_floraison = self.sdd_post_floraison + temperature if self.somfloraison >= 1 else 0 
        
        self.recolte = 1 if self.sdd_post_floraison >= self.sdd_fh else 0 
        self.som_recolte = self.som_recolte + 1 if self.recolte == 1 else 0
    
            
    def update_biomass_and_allocation(self, temperature: float, surface_area: float) -> None:
        if self.recolte == 1:
            self.ban_biomass = 0.0
            self.veg_biomass = 0.0
            self.bun_biomass = 0.0
            self.laiban = 0.0
            
        # FIX: NDD is fixed at floral induction
        self.ndd = self.andd * self.ban_biomass + self.bndd if self.sominiflo == 1 else self.ndd
        self.dmfruit = self.bun_biomass/self.ndd if (self.somfloraison >= 1 and self.recolte < 1 and self.ndd > 0) else 0.0
        
        if self.sominiflo < 1 or self.recolte >= 1: 
            alloc_bun = 0.0
        elif self.sominiflo >= 1 and self.somfloraison < 1:
            alloc_bun = self.Bunchflo / self.sdd_iff * temperature
        else:
            alloc_bun = self.RGR  * temperature * (1 - (self.dmfruit / self.DMfruitmax)) * self.ndd
        
        alloc_bun = min(alloc_bun, self.dDMBAN) 
        
        if self.sominiflo < 1:
            alloc_veg = self.dDMBAN + self.received_biomass 
        else:
            alloc_veg = self.dDMBAN - alloc_bun + self.received_biomass    
        
        self.received_biomass = 0.0
        
        self.ban_biomass += self.dDMBAN  
        self.veg_biomass += alloc_veg + self.alloc_suc 
        self.bun_biomass += alloc_bun 
        
        # FIX: Correct senescence rate to 0.025
        senBan = 0.025 if self.somfloraison >= 1 else 0.013
        plv = 0.51 if self.sominiflo < 1 else 0.3
                      
        if self.somfloraison < 1:
            prod1 = alloc_veg * plv * self.slban * ((self.laiban_max - self.laiban) / self.laiban_max) / surface_area 
            self.laiban = self.laiban + prod1  - (self.laiban * senBan)
        else:
            self.laiban = self.laiban - (self.laiban * senBan)
            
    
    def calculate_mineralN_fromBANresidues(self) -> None:
        """
        Calculate mineral Nitrogen generation from banana plant residues after harvest.
        """
        if self.som_recolte == 1:
            self.Cr0BAN = (self.veg_biomass + self.bun_biomass * self.phampe) * self.pc_ban 
        
        # 2. Process decomposition for week 1 and onwards (>= 1)
        if self.som_recolte >= 1:
            
            # Call the shared engine
            results = nitrogen_release(
                Cr0=self.Cr0BAN, 
                r=self.r_ban, 
                Y=self.residue_c_yield, 
                L=self.bm_decomr, 
                t=self.som_recolte, 
                h=self.h_ban, 
                wr=self.wr_ban, 
                wb=self.wb_ban
            )
            
            # Map the results back to the object's state
            self.CrBAN = results["cr"]
            self.CbBAN = results["cb"]
            self.ChBAN = results["ch"]
            
            self.dCrBAN = results["dCr"]
            self.dCbBAN = results["dCb"]
            self.dchumban = results["dChum"]
            
            self.dNrBAN = results["dNr"]
            self.dNbBAN = results["dNb"]
            self.dnhumban = results["dNhum"]
            
            self.dNRESBAN = results["dNres"]   
            

