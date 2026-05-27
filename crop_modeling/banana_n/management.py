from typing import Dict, List, Sequence

import numpy as np
from typing import Union, Sequence, Dict, Any, List



def banana_fertilizer_schedule(fertilizerdata: List[Dict[str, Any]] = None, nbweeks: int = 40) -> List[Dict]:
    ferti_organizer = BananaNFertiSchedule(nbweeks)
    
    if fertilizerdata:
        ferti_organizer.add_mineral_ferti_events(fertilizerdata)

    return ferti_organizer.ferti_schedule


class BananaNFertiSchedule:
    """
    Organizer to create nitrogen fertilization schedules for banana planting cycles.

    Parameters
    ----------
    total_cycle : int
        Total number of weeks in the banana crop cycle.

    Attributes
    ----------
    nbweeks : int
        Total number of weeks in the crop cycle.
    _fert_template : List[Dict[str, Any]]
        Internal list of weekly fertilization event dictionaries.

    Examples
    --------
    >>> schedule = BananaNFertiSchedule(total_cycle=52)
    >>> schedule.add_ferti_event(n_week_afterplt=4, n_amount=10.5)
    >>> schedule.mineral_fertilize_schedule([
    ...     {"n_week_afterplt": 8, "n_amount": 12.0},
    ...     {"n_week_afterplt": 16, "n_amount": 15.0},
    ... ])
    >>> schedule.ferti_schedule[4]["min_f"]
    10.5
    """

    def __init__(self, total_cycle: int) -> None:
        """
        Initialize the fertilization schedule organizer.

        Parameters
        ----------
        total_cycle : int
            Total number of weeks in the banana crop cycle.

        Raises
        ------
        ValueError
            If `total_cycle` is not a positive integer.
        """
        if not isinstance(total_cycle, int) or total_cycle <= 0:
            raise ValueError(
                f"`total_cycle` must be a positive integer, got {total_cycle!r}."
            )
        self.nbweeks: int = total_cycle
        self._fert_template: List[Dict[str, Any]] = []
        self.restart_template()

    @property
    def ferti_schedule(self) -> List[Dict[str, Any]]:
        """
        Return the current fertilization schedule.

        Returns
        -------
        List[Dict[str, Any]]
            A list of weekly fertilization event dictionaries, one per week.
        """
        return self._fert_template

    @property
    def ferti_template(self) -> Dict[str, Any]:
        """
        Return a blank fertilization event template dictionary.

        This property generates a fresh template on every access and is
        intended as a structural reference used internally by
        `restart_template`.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys:
            - ``"application"`` : bool — whether a fertilization event occurs.
            - ``"week"``        : int  — week index (0-based).
            - ``"q_org"``       : float — organic fertilizer quantity.
            - ``"min_f"``       : float — mineral (nitrogen) fertilizer amount.
        """
        return {
            "application": [],
            "week": [],
            "q_org": [],
            "min_f": [],
        }

    def restart_template(self) -> None:
        """
        Reset the internal fertilization schedule to a blank state.

        All weekly entries are re-initialized: ``"application"`` is set to
        ``False`` and all numeric fields are set to ``0.0``. The ``"week"``
        field is set to the 0-based week index.

        Notes
        -----
        This method is called automatically during ``__init__``. Call it
        manually to clear any previously added fertilization events.
        """
        template_keys = self.ferti_template  # single property access
        self._fert_template = [
            {
                **{
                    k: (False if k == "application" else 0.0)
                    for k in template_keys
                },
                "week": week_idx,
            }
            for week_idx in range(self.nbweeks)
        ]

    def add_ferti_event(self, n_week_afterplt: int, n_amount: float) -> None:
        """
        Add a single nitrogen fertilization event to the schedule.

        Parameters
        ----------
        n_week_afterplt : int
            Week index (0-based, relative to planting) at which the
            fertilization event should occur. Must be within the crop cycle.
        n_amount : float
            Amount of nitrogen (N) to apply at this event.

        Raises
        ------
        ValueError
            If `n_week_afterplt` is outside the valid range
            ``[0, nbweeks - 1]``.

        Examples
        --------
        >>> schedule = BananaNFertiSchedule(total_cycle=52)
        >>> schedule.add_ferti_event(n_week_afterplt=4, n_amount=10.5)
        >>> schedule.ferti_schedule[4]
        {'application': True, 'week': 4, 'q_org': 0.0, 'min_f': 10.5}
        """
        if not (0 <= n_week_afterplt < self.nbweeks):
            raise ValueError(
                f"'n_week_afterplt' must be in [0, {self.nbweeks - 1}], "
                f"got {n_week_afterplt}."
            )
        self._fert_template[n_week_afterplt]["application"] = True
        self._fert_template[n_week_afterplt]["min_f"] = float(n_amount)

    def add_mineral_ferti_events(
        self, application_list: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple nitrogen fertilization events from a list of event dicts.

        Each dictionary in `application_list` is unpacked as keyword
        arguments into :meth:`add_ferti_event`, so every dict must contain
        exactly the keys ``"n_week_afterplt"`` and ``"n_amount"``.

        Parameters
        ----------
        application_list : List[Dict[str, Any]]
            A list of fertilization event descriptors. Each element must be a
            dict with:

            - ``"n_week_afterplt"`` : int — week index relative to planting.
            - ``"n_amount"``        : float — nitrogen amount for the event.

        Raises
        ------
        ValueError
            If any event's ``"n_week_afterplt"`` value is outside the valid
            range, propagated from :meth:`add_ferti_event`.
        KeyError
            If a dict in `application_list` is missing required keys.

        Examples
        --------
        >>> schedule = BananaNFertiSchedule(total_cycle=52)
        >>> schedule.mineral_fertilize_schedule([
        ...     {"n_week_afterplt": 4,  "n_amount": 10.5},
        ...     {"n_week_afterplt": 12, "n_amount": 14.0},
        ... ])
        """
        for event in application_list:
            self.add_ferti_event(**event)

def nitrogen_release(Cr0: float, r: float, Y: float, L: float, t: float, h: float, wr: float, wb: float, wh: float = 0.1) -> Dict[str, float]:
    """
    STICS crop model equation for the carbon in the residue pool over time.
    Calculates carbon pool states and fluxes based on the initial carbon, 
    the decomposition rate, assimilation yield, microbial death rate, and time.
    
    Parameters
    ----------
    Cr0 : float
        The Initial Carbon dumped on the field (e.g., from chopped banana leaves or fertilizer).
    r : float
        The Decomposition Rate of the raw material.
    Y : float
        The Assimilation Yield. Fraction assimilated by microbes into their bodies.
    L : float
        The Microbial Death Rate.
    t : float
        The Time (in weeks) since the harvest happened or the fertilizer was applied.
    h : float
        The Humification Rate. Fraction of decomposing carbon converted into humus.
    wr : float
        The N:C ratio of the raw material.
    wb : float
        The N:C ratio of the microbial biomass.
    wh : float, optional
        The N:C ratio of the humus. Default is 0.1.

    Returns
    -------
    Dict[str, float]
        A dictionary containing state variables and fluxes for carbon and nitrogen:
        - cb, ch, cr: Carbon in biomassa, humus and raw residue
        - dCb, dChum, dCr: Delta of carbon in biomass, humus and raw residue
        - dNb, dNhum, dNr: Delta of nitrogen in biomass, humus and raw residue
        - dNres: Mineral nitrogen released.
    """
    cr = Cr0 * np.exp(-r * t)
    cb = ((r * Y * Cr0) / (L - r)) * (np.exp(-r * t) - np.exp(-L * t))
    ch = Cr0 * ((Y * h / (L - r)) * (r * np.exp(-L * t) - L * np.exp(-r * t)) + Y * h)
    
    dCh = L * h * cb # Humus formed this week.
    dCr = -r * cr # How much raw residue rots this week. (It is negative because the residue tank is losing mass).
    dCb = r * Y * cr - L * cb  # The change in microbe population this week.
    
    dNr = dCr * wr # Nitrogen leaving the raw residue this week
    dNb = dCb * wb # Nitrogen absorbed or released by microbes this week
    dNhum = dCh * wh # Nitrogen locked into new humus this week
    dNres = - dNr - dNb - dNhum # Mineral nitrogen released into the soil this week from the residue decomposition
    
    return {
        "cr": cr, "cb": cb, "ch": ch,
        "dCr": dCr, "dCb": dCb, "dChum": dCh,
        "dNr": dNr, "dNb": dNb, "dNhum": dNhum,
        "dNres": dNres
    }

class BANANAFerti:
    """
    Manages organic fertilizer application and its subsequent mineralization.

    Attributes
    ----------
    of_type : str
        Type of organic fertilizer ('Abflor', 'compost', 'bagasse', 'Fertisol', 'Vegegwa').
    SOMap : float
        Weeks since application.
    Cr0OF : float
        Initial carbon from organic fertilizer.
    CrOF : float
        Current raw carbon pool from fertilizer.
    CbOF : float
        Carbon in microbial biomass pool from fertilizer.
    ChOF : float
        Carbon in humus pool from fertilizer.
    dNhumOF : float
        Nitrogen humidified this week from fertilizer.
    dNrOF : float
        Nitrogen released from raw fertilizer this week.
    dNbOF : float
        Nitrogen absorbed/released by microbes from fertilizer.
    dNRESOF : float
        Net mineral nitrogen released to soil this week from fertilizer.
    """
    def organic_parameters(self) -> None:
        """Loads specific parameters for the selected organic fertilizer type."""
        o_parameters = {
            "Abflor": {
                "CNROF": 4.0,           # C:N ratio of the fertilizer [cite: 3156]
                "CNBOF": 7.8,           # C:N ratio of zymogenous microbial biomass [cite: 3156]
                "pcORG": 0.3314,        # Fraction of carbon in the fertilizer [cite: 3158]
                "rOF": 0.6,             # Decomposition rate constant [cite: 3158]
                "hOF": 0.4              # Humification rate of microbial biomass [cite: 3158]
            },
            "compost": {
                "CNROF": 11.2,
                "CNBOF": 7.8,
                "pcORG": 0.3406,
                "rOF": 0.04078571,
                "hOF": 0.6894089
            },
            "bagasse": {
                "CNROF": 39.0,
                "CNBOF": 34.6461538,
                "pcORG": 0.4101,
                "rOF": 0.05733333,
                "hOF": -0.0591518
            },
            "Fertisol": {
                "CNROF": 9.0,
                "CNBOF": 7.8,
                "pcORG": 0.36,
                "rOF": 0.03511111,
                "hOF": 0.75
            },
            "Vegegwa": {
                "CNROF": 17.0,
                "CNBOF": 42.2117647,
                "pcORG": 0.289,
                "rOF": 0.04870588,
                "hOF": 0.5306354
            }
        }
        
        for k, v in o_parameters[self.of_type].items():
            self.__setattr__(k, v)
    
    def __init__(self, of_type: str = 'Abflor'):
        """
        Initialize the fertilizer manager.
        
        Parameters
        ----------
        of_type : str, optional
            Type of organic fertilizer, by default 'Abflor'. Options include 
            'Abflor', 'compost', 'bagasse', 'Fertisol', and 'Vegegwa'.
        """
        self.of_type = of_type
        
        self.organic_parameters()
        
        self.SOMap = 0
        self.Cr0OF = 0; self.CrOF = 0; self.CbOF = 0; self.ChOF = 0
        self.dNhumOF = 0; self.dNrOF = 0; self.dNbOF = 0
        self.dNRESOF = 0
        self.wrOF: float = 1.0 / self.CNROF; self.wbOF: float = 1.0 / self.CNBOF
    
    def apply_fertilizer(self, is_applied: bool, of_amount: float, Y: float, L: float, wh: float = 0.1) -> None:
        """
        Process the application and ongoing decomposition of organic fertilizer.

        Parameters
        ----------
        is_applied : bool
            Flag indicating whether organic fertilizer is applied this week.
        of_amount : float
            Amount of organic fertilizer applied.
        Y : float
            Assimilation yield (from plant parameters).
        L : float
            Microbial death rate (from plant parameters).
        wh : float, optional
            N:C ratio of the humus, by default 0.1.
        """

        if is_applied:
            self.SOMap = 0
            self.Cr0OF = of_amount * self.pcORG
        
        if self.Cr0OF > 0:
            self.SOMap += 1
            
            # 3. Call the shared engine
            results = nitrogen_release(
                Cr0=self.Cr0OF, 
                r=self.rOF, 
                Y=Y, 
                L=L, 
                t=self.SOMap, 
                h=self.hOF, 
                wr=self.wrOF, 
                wb=self.wbOF,
                wh=wh
            )
            
            # 4. CRITICAL: Extract both the Mineral N AND the Humus N
            self.dNhumOF = results["dNhum"]  # Goes to Soil SON
            self.dNRESOF = results["dNres"]  # Goes to Soil SMN
            
            # (Optional) Save the rest of the state variables for debugging/tracking
            self.CrOF = results["cr"]
            self.CbOF = results["cb"]
            self.ChOF = results["ch"]
            self.dCrOF = results["dCr"]
            self.dCbOF = results["dCb"]
            self.dChumOF = results["dChum"]
            self.dNrOF = results["dNr"]
            self.dNbOF = results["dNb"]
            
        else:
            # If no fertilizer has been applied yet, ensure fluxes are 0
            self.dNhumOF = 0
            self.dNRESOF = 0