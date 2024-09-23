a = "verschaffen; vorschriften; geltenden; aufpassen; beruhigt; Krankenstand; leider; beruhigend; Arbeitsunfälle; höheren; Fehlzeiten; krankheitsbedingten; eingehalten; bemerkt; Gehörschutz; Gefahrenquellen; Lärmbelastung; Werkshallen; Schwerhörigkeit; entstehen; erhöhten; Schlafstörungen; Geräusche; verfolgen; Auf Dauer; unangenehm; empfinden; dringend; raten; schulen; aufmerksam; Lärmmessungen; verschiedenen; anzuschaffen; durchgeführt; Firmenleitung; Bericht; rechnen; Vorschläge; Entwurf; Mitarbeiterschulung; "
myessay = "str()"
targ = a.split(';')

try:
    with open('./doc/essay.txt', 'r') as file:
        lines = file.readlines()
        elems = str()
        missing_words = str()
        for line in lines:
            elems += line
    for ele in targ:
        if ele.lower() in elems.lower():
            continue
        else:
            missing_words += ele
    print(f"didn't find the element : {missing_words}" )
        
except:
    raise FileExistsError