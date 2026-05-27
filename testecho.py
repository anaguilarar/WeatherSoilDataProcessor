
import sys

def banana_n_fertilizer_check(fert_input):

    fertilizer_schedule = {
            'n_week_afterplt': 0,
            'n_amount': 0
            }
    
    if len(fert_input)==0: return [fertilizer_schedule]
    ferti_events = []
    total_n = 0
        
    for m in range(len(fert_input)):
        fert_item = fert_input[m]
        print(fert_item)
        nweek = int(fert_item.get('banana_monthsaftertransplant', None))
        amount = float(fert_item.get('banana_cantidadkgha', None))
        
        if nweek is None:
            ferti_events.append(fertilizer_schedule)
            continue
        
        nvalue = min(float(fert_item.get('banana_n',0)),100)
        pvalue = min(float(fert_item.get('banana_p',0)),100)
        kvalue = min(float(fert_item.get('banana_k',0)),100)
        
        ntotal = round((nvalue/100) * amount, 3)
        ptotal = round((pvalue/100) * amount, 3)
        ktotal = round((kvalue/100) * amount, 3)
        total_n += ntotal
        ferti_events.append({'n_week_afterplt':nweek, 'n_amount':ntotal})
        
    return ferti_events, total_n

def main():
    import json
    #sentence = '{"resultpath": "/var/www/html/suelosdehonduras.gob.hn/storage/app/public/tmp/tmp13/cropmodel/","flagweather": "1","id_user": "13","crop": "coffee","variety": "sun","aldea": "031501","latitud": "14.798048","longitud": "-87.288973","duration": "","fertilizationdata": "[{\"coffee_monthsaftertransplant\":\"\",\"coffee_fuente\":\"\",\"coffee_n\":\"\",\"coffee_p\":\"\",\"coffee_k\":\"\",\"coffee_cantidadkgha\":\"\"}]" }'
    input_json = json.load(sys.stdin)
    
    print(json.loads(input_json["fertilizationdata"]))
    
    print(len(json.loads(input_json["fertilizationdata"])))
    
    print(banana_n_fertilizer_check(json.loads(input_json["fertilizationdata"])))

if __name__ == '__main__':
    main()