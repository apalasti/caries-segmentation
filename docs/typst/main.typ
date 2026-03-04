// AlgoTel/CoRes submission template
// Documentation: https://github.com/balouf/algotel

#import "@preview/algotel:0.1.0": algotel, qed
#import "@preview/algotel:0.1.0": theorem, lemma, proposition, corollary, definition, remark, example, proof

// Change lang: "fr" to lang: "en" for an English submission.
#show: algotel.with(
  title: [Fogszuvasodás-szegmentáció és detekció],
  short-title: [Orvosi képszegmentáció],
  authors: (
    (name: "anonime (asdf)"),
  ),
  abstract: [
    Orvosi fogszuvasodás-szegmentáció és detekció panorámaröntgen-adatokon.
  ],
  lang: "hu",
)

#set cite(style: "ieee")

= Projektterv
Az alábbiakban bemutatjuk vázlatosan a projekttervet: 
- Modellként az U-Net hálózat egy továbbfejlesztett verzióját használjuk.
- Baseline modellként egy alap U-Net hálót alkalmazunk.
- Adathalmazként a DC1000 #cite(<wang2023multi>) és a gyermekkori panorámaröntgen #cite(<zhang2023children>) adathalmazokat használjuk.
- Kiértékelési metrikáknak a Precision, Recall, F1, illetve IoU metrikákat alkalmazzuk.
//- Cél: legalább egy 0.65 Precision illetve Recall elérése 

= Bevezetés

Az orvosi gyakorlatban a fogszuvasodás detektálása sokszor manuális megfigyeléssel történik, ami szubjektív lehet, és a kialakulóban lévő léziók gyakran figyelmen kívül maradhatnak. Erre az egyik megoldás az automatikus megfigyelés és detektálás panorámaröntgen-felvételek alapján @liu2026deep.


= Adathalmazok
 
/*Az alábbiakban bemutatjuk a legkorszerűbb, mélytanulási modellekhez (például szegmentációhoz és objektumdetektáláshoz) használható publikus adathalmazokat:
*/
Az alábbiakban bemutatjuk a választott DC1000 adathalmazt #cite(<wang2023multi>), valamint a gyermekkori fogászati panorámaröntgen adathalmazt #cite(<zhang2023children>), amelyeket a félév során használtunk.
//+ *DC1000 Adathalmaz* 
  /*- *Tartalom:*  1000 panorámaröntgen-felvétel.*/A DC1000 adathalmaz 597 nagy felbontású panorámaröntgen-képből áll. Mindegyik képhez pixelszintű szegmentációs maszk tartozik a fogszuvasodás jelölésére. A röntgenfelvételek annotálását tapasztalt fogorvosok végezték, a képek klinikai forrásból származnak, ami hozzájárul az adathalmaz megbízhatóságához és szakmai hitelességéhez. Az adatgyűjtés széles populációra terjedt ki, így a minták jelentős diverzitást mutatnak a szuvas léziók méretében, intenzitásában és lokalizációjában #cite(<ghimire2025cnns>).

A gyermekkori fogászati panorámaröntgen adathalmaz (Children's Dental Panoramic Radiographs Dataset) 100 kiváló minőségű gyermekkori (2–13 év közötti) és 2692 felnőtt panorámaröntgen-felvételt tartalmaz. Ez az első olyan nyilvánosan elérhető adathalmaz, amely kifejezetten a gyermekgyógyászati páciensekre fókuszál, pixelszintű szegmentációs maszkokat biztosítva a fogszuvasodás detektálásához, valamint annotációkat egyéb fogászati rendellenességek (például periapikális parodontitis, impaktált fogak) felismeréséhez #cite(<zhang2023children>).
  /*- *Annotációk:**/
 /* Több mint 7500 szuvas lézió pixelszintű szegmentációs maszkja. A szuvasodásokat három súlyossági kategóriába (sekély, közepes, mély) sorolták, így kiválóan alkalmas a pontos határok betanítására. */
/*
+ *PRAD-10K Dataset* #cite(<prad10k2025>)
  - *Tartalom:* 10 000 panorámaröntgen-kép, amely jelenleg a szakirodalom egyik legnagyobb elérhető adatbázisa.
  - *Annotációk:* Részletes pixelszintű címkék multistrukturális szegmentációhoz és specifikus betegség-osztályozáshoz.

+ *Intraoral Caries Dataset* #cite(<intraoral6313>)
  - *Tartalom:* 6313 darab intraorális (szájüregen belüli) fotó.
  - *Annotációk:* Kifejezetten objektumdetektálásra felkészített adatbázis, amely YOLO, COCO és Pascal VOC formátumú határoló dobozokat (bounding boxes) tartalmaz, felgyorsítva a valós idejű modellek integrációját.
*/
== Adatelőkészítési terv
Az alábbiakban felsoroljuk azokat a lépéseket, amelyeket az adatok megfelelő előkészítése érdekében szükséges elvégezni:
- Képek átméretezése
- Normalizálás
- Augmentálás: 
  + Véletlenszerű vízszintes tükrözés
  + Eltolás, skálázás és forgatás
  + Véletlenszerű fényerő- és kontrasztállítás
- Adathalmaz felosztása tanító, teszt és validációs adathalmazra



= Mélytanulási architektúrák

== Detekció és Osztályozás

A detekció olyan feladat a képfeldolgozás területén, hogy egy objektum köré egy dobozt határozunk meg. Az osztályozás a képek vagy képrészletek megfelelő osztályba vagy osztályokba besorolása.

  === ResNet 

A Residual Network (ResNet) a mély konvolúciós hálózatok (CNN) eltűnő gradiens problémáját (vanishing gradient) oldja meg @he2016deep. Ennek alapja a reziduális blokk, amely egy "skip connection" (átugró kapcsolat) segítségével továbbítja a bemenetet a rétegek között.

A matematikai definíció a következő. Legyen $x$ a réteg bemenete, és $cal(F)(x, {W_i})$ a tanulható konvolúciós transzformáció. A kimenet $y$ így írható fel:

$ y = cal(F)(x, {W_i}) + x $

Ha a bemenet és kimenet dimenziója eltér, egy lineáris projekciót ($W_s$) alkalmazunk:

$ y = cal(F)(x, {W_i}) + W_s x $

A hálózat kimenetén általában egy Sigmoid aktivációs függvényt alkalmazunk bináris klasszifikáció (szuvas / nem szuvas) esetén:
$ sigma(z) = 1 / (1 + e^(-z)) $


== Szegmentáció

A szegmentáció célja a kép felosztása elkülönülő régiókra vagy szegmensekre a képen található, vizsgált objektum jellemzői alapján #cite(<Zanini2024>). A mi esetünkben ez azt jelenti, hogy a modellnek a kép egyes pixeleit kell besorolnia, hogy az a szuvas területbe tartozik-e vagy háttér.
  === Kiértékelési metrikák
Az alábbiakban bemutatjuk azokat a kiértékelési metrikákat, amelyeket a szakirodalomban is gyakran használnak a szuvasodás-szegmentálásnál #cite(<Zanini2024>). 
#let TP = "True Positive"
// TP: helyesen előrejelzett pozitív pixelek

#let FP = "False Positive"
// FP: hibásan pozitívnak jelölt pixelek

#let FN = "False Negative"
// FN: kihagyott pozitív pixelek

#let GT = "ground truth"
// GT: valós címkézett pixelek halmaza

#let PRED = "predikció"
// PRED: modell által előrejelzett pixelek halmaza


$
"Precision" = "TP" / ("TP" + "FP"),
$
$
 "Recall" = "TP" / ("TP" + "FN")
$
$
 "F1" = 2 * ("Precision" * "Recall") / ("Precision" + "Recall")
$
A Precision azt mutatja, hogy a modell által pozitívnak jósolt pixelek hány százaléka valóban pozitív. Míg a Recall azt mutatja meg, hogy a modell a valóban pozitív pixelek hányad részét jósolta pozitívnak. Az F1  pontszám pedig a Precision és Recall harmonikus átlaga.

$
" IoU" = "TP" / ("TP" + "FP" + "FN")
$
$
"Dice" = (2 * "TP") / (2 * "TP" + "FP" + "FN")
$
Két széles körben használt átfedés-alapú metrika az Intersection over Union (IoU) és a Dice-koefficiens. Az IoU (más néven Jaccard-index) a prediktált és a valódi maszk metszetének és uniójának arányát méri, míg a Dice-koefficiens a metszet kétszeresét viszonyítja a két maszk uniójának méretéhez képest #cite(<taha2015metrics>).
  === U-Net

A szegmentációs feladatok aranystandardja az orvosi képfeldolgozásban az U-Net @ronneberger2015u. Az architektúra egy kódoló (encoder) és egy dekódoló (decoder) ágból áll, melyeket az eredeti térbeli felbontás megtartása érdekében szimmetrikus "skip connection"-ök kötnek össze.

A szegmentáció betanításához a leggyakrabban használt veszteségfüggvény a *Dice Loss* (Dice-Sørensen koefficiens alapján), amely hatékonyan kezeli a képeken jelenlévő osztály-kiegyensúlyozatlanságot (hiszen a szuvas pixel sokkal kevesebb, mint az egészséges fog vagy háttér pixel).

#definition[Dice Veszteségfüggvény]
Legyen $p_i in [0,1]$ a hálózat által jósolt valószínűség az $i$-edik pixelre, és $g_i in {0,1}$ a valós (ground truth) címke. A Dice veszteség ($cal(L)_"Dice"$) a következőképpen számolható:

$ cal(L)_"Dice" = 1 - (2 sum_(i=1)^N p_i g_i + epsilon) / (sum_(i=1)^N p_i + sum_(i=1)^N g_i + epsilon) $
  = SOTA MODELLEK

  == CariesNet
  
A CariesNet egy mélytanulás-alapú szegmentációs modell, amely többstádiumú szuvas léziók detektálására készült panorámaröntgen-felvételeken. Architektúrája az U-Net struktúrájára épül, amelyet egy teljes skálájú axiális figyelmi (Full-Scale Axial Attention – FSAA) modullal, valamint egy részleges enkóder modullal egészítettek ki a szegmentációs teljesítmény javítása érdekében, különös tekintettel a kisebb kiterjedésű léziókra. A modellt 1159 panorámafelvételből álló adathalmazon tanították, amely összesen 3217 annotált szuvas régiót tartalmazott (kezdeti, középsúlyos és mély szuvasodás). A CariesNet 93,64%-os átlagos Dice-együtthatót és 93,61%-os pontosságot ért el, ezzel felülmúlva az olyan alapmodelleket, mint a U-Net, a DeepLabV3+ és a PraNet. Az FSAA modul különösen a lézióhatárok pontosabb kirajzolását segíti elő, míg a részleges enkóder a magas szintű jellemzők aggregálásával járul hozzá a precízebb szegmentációhoz #cite(<zhu2023cariesnet>).
== CariSeg

A CariSeg négy neurális hálózat integrációján alapul, és 99,42%-os pontosságot (accuracy) ért el a fogszuvasodás detektálásában panorámaröntgen-felvételeken. A rendszer első komponense egy U-Net architektúrán alapuló modell, amely a fogak régióját szegmentálja, majd a felvételt az érdeklődési területre fókuszálva (ROI) kivágja. A második komponens a szuvas léziók szegmentálását végzi egy három architektúrából (U-Net, Feature Pyramid Network és DeepLabV3+) álló ensemble modell (modellegyüttes) segítségével. A fogazonosításhoz két egyesített adathalmazt használtak: a Tufts Dental Database 1000 panorámaröntgen-felvételét, valamint egy további, 116 anonim panorámafelvételből álló adatbázist, amely a Noor Medical Imaging Centerben (Qom) készült. A szuvasodás-szegmentációhoz 150 panorámaröntgen-felvételt tartalmazó adatbázist alkalmaztak, amely az Iuliu Hațieganu Orvosi és Gyógyszerészeti Egyetem Száj- és Állcsontsebészeti, valamint Radiológiai Tanszékéről származik. Az ensemble megközelítés az egyes modellek komplementer erősségeit egyesíti, ezáltal kiemelkedő szegmentációs teljesítményt elérve #cite(<muarginean2024teeth>).

== End-to-end mélytanulás-alapú rendszer fogszuvasodás-szegmentációra
 Ebben a munkában egy U-Net-alapú architektúrát alkalmaztak a léziók pixelenkénti szegmentálására, valamint egy ResNet-50 osztályozó hálózatot a többosztályos besorolásra (nincs szuvasodás, zománcszuvasodás, dentinszuvasodás). A U-Net modell 0,89-es Dice-együtthatót ért el, ami a lézióhatárok precíz meghatározását jelzi. A ResNet-50 osztályozó 93,2%-os összpontosságot mutatott, az egyes kategóriák szerinti pontosság pedig 95% (nincs szuvasodás), 91,1% (zománcszuvasodás) és 90,4% (dentinszuvasodás) volt #cite(<marwaha2025end>).

= MLOps Platform

A projekt során az MLOps (Machine Learning Operations) folyamatok menedzselésére az MLflow #cite(<jena2025mlops>) platformot választottuk. Az MLflow segítségével szisztematikusan követhetjük az egyes tanítási kísérleteket, verziózhatjuk a modelljeinket, és biztosíthatjuk az eredmények reprodukálhatóságát.

A rendszert Kubernetes #cite(<poulton2019kubernetes>) környezetben telepítettük egy bérelt szerveren, ami lehetővé teszi a komponensek skálázhatóságát és a hatékony csapatmunkát. Az MLflow használatát három fő modulra alapoztuk:

1. **MLflow Tracking**: Ezzel naplózzuk az összes kísérletet, beleértve a hiperparamétereket (például tanulási ráta, kötegméret), a kiértékelési metrikákat (Dice-együttható, IoU, Precision, Recall) és a tanítás során generált vizualizációkat (például veszteségfüggvény-görbék). Ez kritikus fontosságú a különböző U-Net variánsok és a baseline modell szisztematikus összehasonlításakor.

2. **MLflow Model Registry**: Itt tároljuk a betanított modellek súlyait és metaadatait. A registry segítségével kezeljük a modellek életciklusát, biztosítva, hogy a tesztelés és az alkalmazás során mindig a legmegfelelőbb modellverziót használjuk.

3. **MLflow Artifact Store**: A kísérletek során keletkezett fájlokat, például szegmentációs mintákat és modellsúlyokat központilag, távolról is elérhető módon tároljuk.

Az infrastruktúra alapját képező Kubernetes környezet biztosítja az egyes komponensek (például metaadat-adatbázis, tárolóegység és maga az MLflow szerver) stabil futását és egyszerű üzemeltetését.

A tanításhoz és az interferenciához (inference) GPU-erőforrásokra van szükség, amelyeket az egyetemi klaszter biztosít...

= Eredmények


= Konklúzió

#bibliography("ref.bib", style: "ieee", title: auto)
