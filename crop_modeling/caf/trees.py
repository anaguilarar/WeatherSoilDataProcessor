
class Tree():
    @staticmethod
    def species_params(tree_species):
        def params_1():
            """Erythrina","Erythrina poeppigiana","E. poeppigiana",
               "Poro","poro"""
            CBtree0 = 0.1
            CLtree0 = 0.1
            CRtree0 = 0.1
            CStree0 = 0.1
            FPT = 0
            FST = 0.55
            FWT = 0.83
            HMAX = 6
            KAC = 9.1
            KACEXP = 0.6
            KH = 3.7
            KHEXP = 0.25
            KNFIX = 0.05
            TBEOFREPT = 1825
            TCBT = 2100
            TCRT = 4000
            TCST = 99999
            TOPTT = 24.4
            TTOLT = 8
            return locals()
    
        def params_2():
            """Inga","Inga sp."""
            CBtree0 = 0.1
            CLtree0 = 0.1
            CRtree0 = 0.1
            CStree0 = 0.1
            FPT = 0
            FST = 0.42
            FWT = 0.65
            HMAX = 10
            KAC = 6
            KACEXP = 0.5
            KH = 5
            KHEXP = 0.25
            KNFIX = 0.05
            TBEOFREPT = 1825
            TCBT = 1100
            TCRT = 3400
            TCST = 99999
            TOPTT = 25.9
            TTOLT = 8
            return locals()
        def params_3():
            """Banana", "banana",
                            "Musa sp.","Musa"""
            CBtree0 = 1
            CLtree0 = 1
            CRtree0 = 1
            CStree0 = 1
            FPT = 0.5
            FST = 0.5
            FWT = 0.5
            HMAX = 6
            KAC = 7
            KACEXP = 0.6
            KH = 4.5
            KHEXP = 0.42
            KNFIX = 0
            TBEOFREPT = 100
            TCBT = 365
            TCRT = 365
            TCST = 365
            TOPTT = 22
            TTOLT = 8
            return locals()
        def params_4():
            """Avocado", "avocado",
                            "Persea americana","P. americana"""
            CBtree0 = 0.1
            CLtree0 = 0.1
            CRtree0 = 0.1
            CStree0 = 0.1
            FPT = 0.5
            FST = 0.5
            FWT = 0.5
            HMAX = 6
            KAC = 4
            KACEXP = 0.6
            KH = 3
            KHEXP = 0.4
            KNFIX = 0
            TBEOFREPT = 1825
            TCBT = 1000
            TCRT = 1000
            TCST = 1000
            TOPTT = 20
            TTOLT = 8
            return locals()
        
        def params_5():
            """Grevillea", "G. robusta"""
            CBtree0 = 0.1
            CLtree0 = 0.1
            CRtree0 = 0.1
            CStree0 = 0.1
            FLTMAX = 0.27
            FPT = 0
            FST = 0.25
            FWT = 0.43
            HMAX = 50
            KAC = 6.7
            KACEXP = 0.76
            KH = 5
            KHEXP = 0.25
            KNFIX = 0
            LAIMAXT = 5
            SLAT = 25
            TBEOFREPT = 1825
            TCBT = 2600
            TCRT = 5200
            TCST = 99999
            TOPTT = 25.0
            TTOLT = 8
            return locals()
        
        def params_6():
            """Cordia", "C. alliodora"""
            CBtree0 = 0.1
            CLtree0 = 0.1
            CRtree0 = 0.1
            CStree0 = 0.1
            FLTMAX = 0.24
            FPT = 0
            FST = 0.3
            FWT = 0.55
            HMAX = 50
            KAC = 6.7
            KACEXP = 0.84
            KH = 5.6
            KHEXP = 0.33
            KNFIX = 0
            LAIMAXT = 5
            SLAT = 25
            TBEOFREPT = 1825
            TCBT = 2600
            TCRT = 5200
            TCST = 99999
            TOPTT = 25.0
            TTOLT = 8
            return locals()
        
        print(f"Tree parameters set for {tree_species.lower()}")
        
        if tree_species.lower() in ["erythrina","erythrina poeppigiana", "poro"]:
            return params_1()
        elif tree_species in ["inga","inga sp."]:
            return params_2()
        elif tree_species.lower() in ["banana", "musa sp.","musa"]:
            return params_3()
        elif tree_species.lower() in ["avocado", "Persea americana"]:
            return params_4()
        elif tree_species.lower() in ["grevillea", "g. robusta"]:
            return params_5()
        elif tree_species.lower() in ["cordia", "c. alliodora"]:
            return params_6()
        elif tree_species.lower() in ["sun"]:
            return params_1()
        else:
            raise ValueError(f'there are not parameters for {tree_species}')

    