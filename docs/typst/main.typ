// AlgoTel/CoRes submission template
// Documentation: https://github.com/balouf/algotel

#import "@preview/algotel:0.1.0": algotel, qed
#import "@preview/algotel:0.1.0": theorem, lemma, proposition, corollary, definition, remark, example, proof

// Change lang: "fr" to lang: "en" for an English submission.
#show: algotel.with(
  title: [Fogszuvasodás Szegmentáció és Detekció],
  short-title: [Orvosi kép szegmentáció],
  authors: (
    (name: "name"),
    (name: "name")
  ),
  abstract: [
    Orvosi fogszuvasodás-szegmentáció és detekció panorámaröntgen-adatokon.
  ],
  lang: "hu",
)

#set cite(style: "ieee")
#set text(lang: "hu")
#show figure.where(kind: image): set figure(supplement: [Ábra])

= Projektterv
Az alábbiakban bemutatjuk vázlatosan a projekttervet 
- Detekcióhoz egy YOLO-stílusú modellt használunk
- Szegmentációhoz U-Net modellt használunk
- A két komponenst YOLO + U-Net konjunkciós architektúrában kapcsoljuk össze
- Adathalmaznak a DC1000 adathalmazt használjuk
- Kiértékelési metrikáknak a Precision, Recall, F1 illetve IoU metrikákat használjuk
//- Cél: legalább egy 0.65 Precision illetve Recall elérése 

== Modellterv

A modellek fejlesztését négy egymásra épülő lépésben tervezzük:

1. Detektor alapmodell (YOLO): fog-régiók detektálása bounding box szinten.
2. Szegmentáló alapmodell (U-Net): pixel-szintű szuvasodás maszkok tanítása.
3. Konjunkciós modell (YOLO + U-Net): a YOLO által detektált régiók kivágása, majd lokális maszkfinomítás U-Net segítségével.
4. Összehasonlítás és ablatív vizsgálat: külön értékeljük a YOLO-only, U-Net-only és a konjunkciós pipeline teljesítményét.

= Bevezetés

Az orvosi felhasználásban a fog szuvasodás sokszor manuális megfigyeléssel történik, ami sokszor szubjektív és a kialakulóban lévő fog szuvasodást sokszor figyelmen kívül marad. Erre az egyik megoldás az automatikus megfigyelés panoráma röntgen felvétel alapján @liu2026deep.


= Adathalmazok
 
/*Az alábbiakban bemutatjuk a legkorszerűbb, mélytanulási modellekhez (például szegmentációhoz és objektumdetektáláshoz) használható publikus adathalmazokat:
*/
Az alábbiakban bemutatjuk a választott DC1000 adathalmazt #cite(<wang2023multi>), amelyet a félév során használtunk.
//+ *DC1000 Adathalmaz* 
  /*- *Tartalom:*  1000 panoráma röntgenfelvétel.*/A DC1000 adathalmaz 597 nagy felbontású panoráma röntgenképből áll. Mindegyik képhez pixel-szintű szegmentációs maszk tartozik a fogszuvasodás jelölésére. A röntgenfelvételek annotálását tapasztalt fogorvosok végezték, a képek klinikai forrásból származnak, ami hozzájárul az adathalmaz megbízhatóságához és szakmai hitelességéhez. Az adatgyűjtés széles populációra terjedt ki, így a minták jelentős diverzitást mutatnak a szuvas léziók méretében, intenzitásában és lokalizációjában #cite(<ghimire2025cnns>).
  /*- *Annotációk:**/
 /* Több mint 7500 szuvas lézió pixelszintű szegmentációs maszkja. A szuvasodásokat három súlyossági kategóriába (sekély, közepes, mély) sorolták, így kiválóan alkalmas a pontos határok betanítására. */
/*
+ *PRAD-10K Dataset* #cite(<prad10k2025>)
  - *Tartalom:* 10 000 panoráma röntgenkép, amely jelenleg a szakirodalom egyik legnagyobb elérhető adatbázisa.
  - *Annotációk:* Részletes pixel-szintű címkék multi-strukturális szegmentációhoz és specifikus betegség-osztályozáshoz.

+ *Intraoral Caries Dataset* #cite(<intraoral6313>)
  - *Tartalom:* 6313 darab intraorális (szájüregen belüli) fotó.
  - *Annotációk:* Kifejezetten objektumdetektálásra felkészített adatbázis, amely YOLO, COCO és Pascal VOC formátumú határoló dobozokat (bounding boxes) tartalmaz, felgyorsítva a valós idejű modellek integrációját.
*/

= Adatelőkészítés


A Roboflow-train adatkészlet esetében gyakran előfordultak azonos képek különböző augmentált változatai. Ezeket csoportosítottuk az azonos alapnév alapján, és minden csoportból csak egy példányt tartottunk meg, míg a többi képet és a hozzá tartozó címke fájlokat eltávolítottuk. Ezzel a lépéssel biztosítottuk, hogy a tanító adatkészlet ne tartalmazzon redundáns példákat, ami a modell tanulási folyamatát torzíthatná.

A különböző forrásokból származó képek maszkjainak előállítása is eltérő módszert igényelt. A DC1000 képeknél a megfelelő maszkokat a képekhez tartozó színes annotációs mappákból nyertük ki, míg a Roboflow képeknél a YOLO formátumú címkék alapján hoztunk létre pixelmaszkokat. Ehhez a poligon koordinátákat átskáláztuk a kép méretének megfelelően, és a pixelértékekkel reprezentált maszkot hoztunk létre, ahol a háttér 0, a szuvas terület pedig 255 értéket kapott. Így minden képhez elkészült a modell tanításához szükséges, egységes formátumú, pixelalapú maszk.

Az előkészített adatkészletet ezután determinisztikus módon osztottuk szét tanító, validációs és teszt részhalmazokra. Minden képet egy egyedi azonosítóval láttunk el, amely biztosította, hogy ugyanaz a kép mindig ugyanabba a részhalmazba kerüljön.

A feldolgozás során azokat a képeket, amelyekhez nem állt rendelkezésre megfelelő maszk vagy feldolgozásuk során hiba lépett fel, kidobtuk, így a végső adatkészlet csak teljesen feldolgozható, minőségi példákat tartalmazott.

A tanító, validációs és teszt adatkészletek előfeldolgozott képpárokból állnak, ahol minden képhez egy bináris maszk tartozik a szuvas területek jelölésére. Az adatok 256×256 méretre vannak átméretezve és normalizálva. A különböző források (DC1000 és Roboflow) kontrolláltan kerülnek a tanító, validációs és teszt halmazokba, a reprodukálhatóság érdekében determinisztikus split-et alkalmaztunk. A SegmentationDataModule biztosítja a batch-ek konzisztens betöltését és a shuffle lehetőséget a tanításhoz, míg a BaseKariesDataset kezeli az egyes képpárokat, és a maszkok bináris ábrázolását.

== Online Augmentáció

A tanító adatokon online augmentációt alkalmazunk: a minták betöltésekor
minden alkalommal (epochonként és batch-enként) véletlenszerűen transzformáljuk a
képet és a hozzá tartozó szegmentációs maszkot, hogy a hálózat nagyobb
változatosságot lásson. Az implementáció az Albumentations könyvtárra épül, a
lépések egy összefűzött láncban futnak, előre rögzített sorrendben.

+ Vízszintes tükrözés: bal-jobb tükrözés; a maszk ugyanazzal a geometriával transzformálódik, mint a kép
+ Véletlenszerű méretarány: tisztán geometriai változatosságot ad (eltolás és forgatás nélkül)
+ Véletlenszerű fényerő és kontraszt:

== Validáció

Validáción (és tipikusan teszten) nem használunk véletlen augmentációt: csak
az átméretezés történik ugyanarra a célméretre.


= Adatvizualizáció

== Adathalmaz bemutatása

Az adathalmaz felosztását az alábbi ábra szemlélteti, amely megmutatja a tanító, validációs és teszt halmaz arányát.

#figure(
  image("./figures/dataset_introduction/dataset_split_pie.png", width: 60%),
  caption: [Az adathalmaz felosztása tanító, validációs és teszt halmazokra.]
)

Továbbá az alábbi ábra szemlélteti hogy a két felhasznált adathalmaz (Roboflow, DC1000) esetében az egyes adat szegmensek milyen arányban tartalamznak képeket a két említett adathalmazból.

#figure(
  grid(
    columns: 3,
    gutter: 10pt,

    [
      #image("./figures/dataset_introduction/source_distribution_train.png", width: 100%)
    ],

    [
      #image("./figures/dataset_introduction/source_distribution_val.png", width: 100%)
    ],

    [
      #image("./figures/dataset_introduction/source_distribution_test.png", width: 100%)
    ],
  ),
  caption: [Az osztályeloszlás összehasonlítása a tanító, validációs és teszt halmazokban.]
)

== Annotáció vizualizáció

Az alábbi két ábra a Roboflow és a DC1000 adathalmazból származó képek annotációit mutatja. Az eredeti képek mellett látható a bináris maszk és az overlay, amely vizuálisan kiemeli a szuvas területeket.

#figure(  image("./figures/annotations/annotation_examples_roboflow.png", width: 60%),
  caption: [Példák a roboflow adathalmazból az eredeti, maszkolt, illetve átfedésre (overlay)]
)
#figure(
  image("./figures/annotations/annotation_examples_DC1000.png", width: 60%),
  caption: [Példák a DC1000 adathalmazból az eredeti, maszkolt, illetve átfedésre (overlay)]
)





= Online Augmentáció

A tanító adatokon *online augmentációt* alkalmazunk: a minták betöltésekor
minden alkalommal (epochonként és batch-enként) véletlenszerűen transzformáljuk a
képet és a hozzá tartozó szegmentációs maszkot, hogy a hálózat nagyobb
változatosságot lásson. Az implementáció az `Albumentations` könyvtárra épül, a
lépések egy összefűzött láncban futnak, előre rögzített sorrendben.

+ *Vízszintes tükrözés*: bal-jobb tükrözés; a maszk ugyanazzal a geometriával transzformálódik, mint a kép
+ *Véletlenszerű méretarány*: tisztán geometriai változatosságot ad (eltolás és forgatás nélkül)
+ *Véletlenszerű fényerő és kontraszt*:

== Validáció

Validáción (és tipikusan teszten) *nem* használunk véletlen augmentációt: csak
az átméretezés történik ugyanarra a célméretre.

== Szegmentáció

A szegmentáció célja a kép felosztása elkülönülő régiókra vagy szegmensekre a képen található, vizsgált objektum jellemzői alapján #cite(<Zanini2024>). A mi esetünkben ez azt jelenti, hogy a modellnek a kép egyes pixeleit kell besorolnia, hogy az szuvas területbe számít-e bele vagy háttérnek.
  === Kiértékelési metrikák
Az alábbiakban bemutatjuk azokat a kiértékelési metrikákat, amelyeket a szakirodalomban is gyakran használnak szuvasodás szegmentálásánál #cite(<Zanini2024>).
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
Két széles körben használt átfedés-alapú metrika az Intersection over Union (IoU) és a Dice-koefficiens. Az IoU (más néven Jaccard-index) a prediktált és a valódi maszk metszetének és uniójának arányát méri, míg a Dice-koefficiens a metszetnek a kétszeresét viszonyítja a két maszk úniójának méretéhez képest #cite(<taha2015metrics>).

= Mélytanulási architektúrák

== U-Net

A szegmentációs feladatok aranystandardja az orvosi képfeldolgozásban az U-Net @ronneberger2015u. Az architektúra (#ref(<fig-cimke>)) egy kódoló (encoder) és egy dekódoló (decoder) ágból áll, melyeket az eredeti térbeli felbontás megtartása érdekében szimmetrikus "skip connection"-ök kötnek össze.

#figure(
  image("u-net.png", width: 80%),
  caption: [U-net architekrúrája.],
) <fig-cimke>

A szegmentáció betanításához a leggyakrabban használt veszteségfüggvény a *Dice Loss* (Dice-Sørensen koefficiens alapján), amely hatékonyan kezeli a képeken jelenlévő osztály-kiegyensúlyozatlanságot (hiszen a szuvas pixel sokkal kevesebb, mint az egészséges fog vagy háttér pixel).

#definition[Dice Veszteségfüggvény]
Legyen $p_i in [0,1]$ a hálózat által jósolt valószínűség az $i$-edik pixelre, és $g_i in {0,1}$ a valós (ground truth) címke. A Dice veszteség ($cal(L)_"Dice"$) a következőképpen számolható:

$ cal(L)_"Dice" = 1 - (2 sum_(i=1)^N p_i g_i + epsilon) / (sum_(i=1)^N p_i + sum_(i=1)^N g_i + epsilon) $

=== U-Net Architektúra Kiterjesztése és Implementációs Részletek

Jelen kutatásban az eredeti U-Net modellt vettük alapul, amelyet egy mélyebb, Large U-Net architektúrára is kiterjesztettünk, hogy a rendelkezésre álló GPU (NVIDIA RTX) memóriáját hatékonyabban használjuk ki, illetve robusztusabb reprezentációt tanulhassunk a röntgenfelvételekből. Az implementáció a Python programozási nyelven, a PyTorch keretrendszer #cite(<paszke2019pytorch>) segítségével készült.

==== Konvolúciós Blokk, Kernel, Padding és Stride

Az architektúra alapköve a duplázott konvolúciós blokk (`DoubleConv`), amely egyaránt alkalmazásra kerül az "encoder" és a "decoder" ágban. Ez a blokk két egymást követő kétdimenziós konvolúciós rétegből (`Conv2d`) áll. A konvolúciós szűrők (kernel) mérete $3 times 3$, amely standard értékként elegendő a lokális térbeli mintázatok és élek felismeréséhez.


Annak érdekében, hogy a transzformáció során a hálózat feleslegesen ne csökkentse az aktivációs térképek térbeli felbontását (magasságát és szélességét), egységnyi kitöltést (`padding=1`) alkalmaztunk. A lépésköz (`stride`) értéke a konvolúciós blokk belsejében $1$, így a konvolúciós ablak minden egyes pixelre finoman rácsúszik, megőrizve a rácsfelbontást.


A kódoló szakaszban a térbeli redukciót mindig hálózaton kívüli, exkluzív $2 times 2$-es Max Pooling réteg végzi (ahol a lépésköz is 2), ami rendre megfelezi a felépített reprezentációk felbontását. A dekódoló ág feladata ennek ellentéte, a térbeli dimenziók visszaállítása transzponált konvolúciók (`ConvTranspose2d`) segítségével ($2 times 2$-es kernel, $2$-es stride kíséretében).




==== Batch Normalization

Minden konvolúciós műveletet a PyTorch `BatchNorm2d` rétege követi a
nem-lineáris aktivációs függvény (ReLU - Rectified Linear Unit) előtt. A Batch
Normalizáció #cite(<ioffe2015batch>) megkönnyíti és felgyorsítja a mély neurális
hálózatok betanítását azzal, hogy az egyes rétegek bemeneteinek eloszlását
fixálja, redukálva az ún. belső kovariancia eltolódás (internal covariate shift)
jelenségét.


Matematikailag a normálás a következőképpen történik az adott mini-batch-en
belül minden csatornára függetlenül:


$ hat(x)_i = (x_i - mu_"batch") / sqrt(sigma_"batch"^2 + epsilon) $
$ y_i = gamma hat(x)_i + beta $

ahol $mu_"batch"$ a mini-batch adott térképre vonatkozó empirikus átlaga, $sigma_"batch"^2$ a varianciája, a $gamma$ (skála) és $beta$ (eltolás) pedig a hálózat által tanult paraméterek. Ennek következtében a PyTorch stabilabb gradiensáramlást tud produkálni végig az egész U-Net testen keresztül.

== YOLO és U-net architektúra terv:
A projectben YOLOv5 és U-Net architektúrákat
használunk a detekció és szegmentáció feladatokhoz. A két modellt egy
konjunkciós architektúrában kapcsoljuk össze, ahol a YOLO detektor azonosítja a
fogakat, majd az U-Net ezeken a régiókon szegmentációt végez.
\
\
A YOLO detektor fő komponensei:
- *Backbone:* egymásra épülő konvolúciós blokkok (`Conv2d + BatchNorm2d + SiLU`), amelyek többlépcsős leskálázással robusztus jellemzőtérképet építenek.
- *Detekciós fej (detection head):* $1 times 1$ konvolúció, amely horgonypontonként (anchor) a következőket becsli: dobozparaméterek $(x, y, w, h)$, objektumosság és osztályvalószínűség.
- *Dekódolás és szűrés:* sigmoid aktiváció, rácskoordináta-alapú visszaskálázás, majd Non-Maximum Suppression (NMS) a duplikált dobozok eltávolítására.

A detektor kimeneti csatornaszáma a következő:

$ C_"out" = A * (5 + C) $

ahol $A$ az anchorok száma, $C$ pedig az osztályok száma. A képletben szereplő $5$ a YOLO doboz-leírás fix komponenseit jelenti: $(x, y, w, h)$ koordinátaparaméterek + objektumossági pontszám ("obj"ectness) @redmon2016yolo.
\
\
A jelenlegi implementációban nem használunk közös, end-to-end kombinált
veszteségfüggvényt a YOLO és az U-Net között. A két modell külön lépésben tanul:
először a YOLO detektor, majd külön az U-Net a YOLO által kijelölt régiókon.
\
\
Az új architekturális elem a *YOLO + U-Net konjunkciós blokk*:
- a YOLO által detektált bounding box régiókat kivágjuk,
- opcionális paddinget adunk a kontextus megőrzésére,
- a kivágást fix U-Net bemeneti méretre mintavételezzük,
- az U-Net lokális szegmentációt készít,
- a bináris maszkot visszavetítjük az eredeti képre és régiónként egyesítjük.

Ezzel a felépítéssel a YOLO biztosítja a gyors régió-jelölést, míg az U-Net a pixelek szintjén pontosítja a szuvas területek határát.

  = Gépi tanulás modell bemutatás
== Implementációs részletek

=== Kombinált veszteségfüggvény

A szegmentációs modell tanítása során egy kombinált veszteségfüggvényt alkalmazunk, amely két komponenst tartalmaz: a bináris keresztentrópia ("BCE") veszteséget és a Dice-alapú átfedési veszteséget.

A teljes veszteség:

$
L = L_"BCE" + L_"Dice"
$

ahol: $L_"BCE"$: pixel-szintű bináris keresztentrópia veszteség, $L_"Dice"$: átfedés-alapú Dice veszteség

==== Bináris keresztentrópia veszteség

A "BCE" veszteség a logit-alapú előrejelzések és a ground truth maszkok közötti különbséget méri:

$
L_"BCE" = "BCEWithLogits"(p, y)
$

ahol:  $p$: a hálózat logit kimenete, $y$: bináris ground truth maszk.

==== Dice veszteség

A Dice veszteség az átfedés minőségét méri:

$
L_"Dice" = 1 - frac(2 * |P ∩ Y|, |P| + |Y|)
$

ahol: $P$: predikált maszk $Y$: ground truth maszk

==== Összesített veszteség

A két komponens összege:

$
L = "BCEWithLogits"(p, y) + L_"Dice"(p, y)
$

==== Súlyozás
Az alábbi képlet mutatja hogyan történik a dice loss súlyozása.
$
L_"Dice"^"weighted" = sum_i w_i * L_"Dice"^(i)
$
ahol a $w_i$ súlyok az egyes osztályok fontosságát
szabályozzák.

=== Tanulási ráta ütemezés (Learning Rate Scheduler)

A modell tanítása során adaptív tanulási ráta ütemezést alkalmazunk a konvergencia stabilizálása érdekében. A használt stratégia a "ReduceLROnPlateau" scheduler, amely a validációs teljesítmény alapján csökkenti a tanulási rátát.

A scheduler a következő metrikát figyeli:

$
m = "val/dice"
$

amely a validációs Dice pontszámot jelöli.

=== Működési elv

Amennyiben a figyelt metrika nem javul egy adott számú epoch-on keresztül, a tanulási ráta csökkentésre kerül:

$
lr = lr * gamma
$

ahol:
- $gamma = 0.5$: csökkentési faktor
- $"patience" = 5$: türelmi periódus epochokban




A scheduler célja, hogy:
- gyors kezdeti tanulást biztosítson magas tanulási rátával
- majd finomhangolást végezzen kisebb tanulási rátával
- elkerülje a lokális minimumokban való beragadást

=== Formális feltétel

A tanulási ráta csökkentése akkor történik, ha:

$
m_t <= max(m_{t-"patience"}, ..., m_{t-1})
$

azaz a metrika nem mutat javulást a megadott türelmi ablakban.

=== Implementációs megjegyzés

A scheduler epoch végén frissül, és a validációs Dice pontszám alapján kerül meghívásra:

- monitor: "val/dice"
- mode: "max"
- interval: epoch

//todo ide kod + elozo fejezet alapjan leirni a vegleges modell felepitest, mukodest



== YOLOv8-alapú fogdetektáló modell működése és eredményei

A detekciós fázisban az Ultralytics YOLOv8 architektúrát alkalmaztuk a fogak lokalizálására. A modell feladata a releváns régiók (ROI) kijelölése, amelyeken később a szegmentációt végezzük.

=== 4-fold Keresztvalidációs Tanítás

A modell robusztusságának biztosítása érdekében 4-fold keresztvalidációs (cross-validation) eljárást alkalmaztunk. A teljes adathalmazt 4 diszjunkt részre osztottuk, és minden foldon külön tanítottuk a modellt 100 epochon keresztül.

A keresztvalidáció során elért eredmények (mAP50-95):
- *Fold 1:* 0.630
- *Fold 2:* 0.642
- *Fold 3:* *0.6543 (Legjobb)*
- *Fold 4:* 0.6507

A **mAP50-95** (mean Average Precision) az objektumdetekció legfontosabb mérőszáma. Azt méri, hogy a modell mennyire találja meg az összes objektumot (Recall), és a jósolt dobozok mennyire pontosan fedik le azokat (Precision). Az "50-95" jelölés azt jelenti, hogy az átlagpontosságot több különböző IoU (Intersection over Union) küszöbértéken (0.5-től 0.95-ig, 0.05-ös lépésközzel) számoljuk ki, ami szigorúbb és pontosabb képet ad a modell lokalizációs képességéről.

A végleges kétfázisú modellbe a legjobb teljesítményt nyújtó, 3. foldból származó súlyokat integráltuk.

=== Veszteségfüggvények

A YOLOv8 architektúra egy összetett veszteségfüggvényt használ, amely a következő komponensekből áll:

1. *Box Loss ($L_"box"$):* A lokalizáció pontosságát méri a *CIoU (Complete Intersection over Union)* metrika segítségével, amely figyelembe veszi az átfedést, a középpontok távolságát és az oldalarányokat.
2. *Classification Loss ($L_"cls"$):* Bináris keresztentrópiát (*BCE*) használ az objektumok (fogak) jelenlétének osztályozására.
3. *Distribution Focal Loss ($L_"dfl"$):* Segíti a bounding box határainak finomhangolását elmosódott szélek esetén.

=== Caries lefedettség a prediktált régiókban

Meghatároztuk, hogy a detektor által jósolt bounding boxok mennyire fedik le a tényleges szuvasodásokat (ground truth). A globális metrikák alapján a teljes szuvas terület eloszlása:
- *Caries terület a BBoxokon belül:* 96.86%
- *Caries terület a BBoxokon kívül:* 3.14%

Ez az eredmény igazolja a detekciós fázis hatékonyságát, mivel a szuvasodások elhanyagolható része marad ki a szegmentálásra kerülő patchekből.

=== Tanítási görbék és példák

#figure(
  image("./figures/training/yolo_training_curves_cv.png", width: 80%),
  caption: [A YOLOv8 4-fold keresztvalidáció tanítási görbéi (loss és mAP értékek).]
)

A vizuális kiértékelés során az alábbi ábrákon direkt olyan példákat mutatunk, ahol a szegmentáció nehézségekbe ütközött, és nem sikerült a caries területét teljes egészében a jósolt bounding boxon belül tartani. Ezek a határesetek rávilágítanak a detektor és a szegmentáló modell közötti interakció kritikus pontjaira.

#figure(
  grid(
    columns: 2,
    gutter: 10pt,
    [ #image("./figures/evaluation/yolo_bbox_predictions_on_caries_1.png", width: 100%) <sample0> ],
    [ #image("./figures/evaluation/yolo_bbox_predictions_on_caries_2.png", width: 100%) <sample1> ]
  ),
  caption: [Példák a YOLOv8 bounding box predikcióira, ahol a szegmentáció részben kívül esett a dobozon.]
)

=== Észrevétel:
A modell **gyakran nem detektálja a fogsor szélén elhelyezkedő fogakat**.

== Kezdetleges eredmények a kétfázisú illetve U-net baseline modllekre

A teszt adathalmazon kiértékeltük a kétfázisú modellt (Yolo+U-net), illetve a baseline U-net modellt is.
A *baseline U-Net modell* esetében a bemeneti képek felbontása 256 × 256 pixel, az előfeldolgozott adatok a data/preprocessed könyvtárból származnak.

*Adataugmentáció:* a tanítás során fényerő- és kontrasztmódosítást alkalmaztunk (p = 0.5, ±0.1 tartomány), rugalmas deformációt (p = 0.3, α = 30, σ = 5), horizontális tükrözést (p = 0.5), skálázást (p = 0.5, ±20%), valamint fókuszált kivágást (p = 0.8).

*Modell:* a szegmentációhoz U-Net architektúrát használtunk, 64 kezdő csatornával és 4 szint mélységgel. A háló dropout regularizációt alkalmaz (p = 0.2). A bemenet egysávos (1 csatorna), a kimenet bináris maszk (1 csatorna).

*Tanítás:* a modellt 500 epochon keresztül tanítottuk, 64-es batch mérettel. Optimalizálóként AdamW optimizer-t alkalmaztunk, 1 × 10⁻⁴ kezdeti tanulási rátával, amely a tanítás során 6.25 × 10⁻⁶ értékre csökkent. A modell 180 epoch után stabil konvergenciát mutatott (gradiens norma ≈ 1.10).

*Veszteségfüggvény:* a tanítás során kombinált Dice és Focal veszteséget alkalmaztunk, kiegészítve egy kisebb súlyú bináris keresztentrópia (BCE) komponenssel. A Focal loss paraméterei α = 0.85 és γ = 2, súlya 0.8, míg a BCE komponens súlya 0.1, amely fokozatosan került bevezetésre az első 50 epoch során. A tanítás végére a teljes veszteség 0.373 értéket vett fel (Dice loss: 0.368, Focal loss: 0.0049).



A teszt adathalmazon kiértékeltük a kétfázisú modellt is, amely egy detekciós (YOLO-alapú) és egy szegmentációs (U-Net) komponens egymásra építésével működik. A modell célja, hogy először lokalizálja a releváns fogterületeket, majd azokon finom szegmentációt végezzen.


A *YOLO + U-Net kétfázisú modell* esetében úgy végeztük a tanítást, hogy előbb a detekciós komponenst (YOLO) tanítottuk meg a fogak lokalizálására, majd a kapott bounding boxok alapján kivágott régiókon külön tanítottuk a szegmentációs (U-Net) modellt. A detektor által előállított régiók köré kis mértékű padding-et alkalmaztunk a kontextus megőrzése érdekében, majd ezeket egységes méretre (256 × 256 pixel) skáláztuk.

A tanítás során a két komponens nem egyszerre, hanem egymást követően került optimalizálásra: először a YOLO modell konvergenciáját biztosítottuk, majd annak kimenetét fixálva tanítottuk a U-Net modellt.

*Adatok:* A bemeneti képek a detektor esetében 640 × 640 pixel felbontásúak, míg a szegmentáló hálózat 256 × 256 pixelre átméretezett kivágásokat (ROI-kat) kap.
A bemenetek normalizálása [0, 1] tartományba történik.

*Adataugmentáció:* A detekciós modell tanítása során tipikus objektumdetekciós augmentációk kerülnek alkalmazásra (pl. skálázás, tükrözés), míg a szegmentációs komponens esetében a baseline U-Net modellhez hasonló augmentációs eljárásokat alkalmaztunk a kivágott régiókon.
A kivágások során 5% padding-et alkalmaztunk a detektált bounding boxok körül a kontextus megőrzése érdekében.

Modell: A kétfázisú architektúra első komponense egy YOLO-alapú objektumdetektor, amely a fogak lokalizálására szolgál. A modell maximum 32 detekciót ad képenként, nem-maximális elnyomást (NMS) alkalmazva 0.55 IoU küszöbbel, valamint 0.22-es konfidencia küszöbbel.

A *szegmentációs (U-Net)* komponens a detektált régiók kivágásain került tanításra. A modell 140 epoch után stabil konvergenciát mutatott, ahol a gradiens norma ≈ 1.26 volt. A tanulási ráta a tanítás során 2.5 × 10⁻⁵ értékre csökkent.

*Tanítás:* A detekciós komponenst (YOLO) 20 epochon keresztül tanítottuk, 8-as batch mérettel. Az optimalizáláshoz Adam-alapú megközelítést alkalmaztunk 5 × 10⁻⁴ kezdeti tanulási rátával és 1 × 10⁻⁵ súlycsökkentéssel (weight decay). A tanítás során nem alkalmaztunk tanulási ráta ütemezést vagy early stopping mechanizmust.

A szegmentációs (U-Net) komponenst nem teljes képeken, hanem patch-alapú megközelítéssel tanítottuk. A bemeneti képeket 128 × 128 méretű kivágásokra bontottuk, és ezeken végeztük a tanítást, amely lehetővé tette a lokális struktúrák részletesebb tanulását. A mintavételezés során 0.8 valószínűséggel fókuszált kivágásokat alkalmaztunk, biztosítva, hogy a releváns régiók nagyobb arányban jelenjenek meg a tanító adatok között.

A U-Net modell 140 epochon keresztül került tanításra, és stabil konvergenciát mutatott (gradiens norma ≈ 1.26). Az optimalizálás AdamW optimizerrel történt, ahol a tanulási ráta a tanítás végére 2.5 × 10⁻⁵ értékre csökkent.

Az end-to-end konfiguráció során a tanítás 0.5 valószínűséggel ground truth bounding boxokat használ a stabilabb tanulás érdekében, csökkentve a detekciós hibák továbbterjedését a szegmentációs komponens felé. A kivágások minimális mérete 2 pixel, valamint további 10 pixeles padding került alkalmazásra a bounding boxokra.

A kiértékelés során a szegmentációs modell teljesítményét nem a teljes képeken, hanem a detektor által meghatározott bounding boxokon belül vizsgáltuk. A detektált régiók kivágásra kerültek, majd ezekre alkalmaztuk a U-Net modellt, és az így kapott maszkokat hasonlítottuk össze a ground truth annotációkkal.

*Veszteségfüggvény*: A detekciós komponens esetében a YOLO architektúrára jellemző összetett veszteségfüggvényt alkalmaztuk, amely tartalmaz lokalizációs, objektumossági és klasszifikációs komponenseket.
 szegmentációs komponens esetében a baseline modellhez hasonlóan kombinált Dice és Focal veszteséget alkalmaztunk, kiegészítve egy kisebb súlyú bináris keresztentrópia (BCE) komponenssel.
 //A végső maszk előállítása 0.5-ös küszöböléssel történik.



#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("./figures/m4_phase/u-net_baseline_loss_curve.png", width: 100%)
    ],

    [
      #image("./figures/m4_phase/2phase_unet_learning_curve.png", width: 100%)
    ]
  ),
  caption: [A U-net baseline (balra) modell illetve a kétfázisú modell (jobboldali) tanulási görbéi a tanító és validációs adathalmazon.]
)

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("./figures/m4_phase/u-net_baseline_confusion_mtix.webp", width: 100%)
    ],

    [
      #image("./figures/m4_phase/2phase_confusion_mtix.webp", width: 100%)
    ]
  ),
  caption: [A U-net baseline (balra) modell illetve a kétfázisú modell (jobboldali) confusion mátrixai a teszt adathalmazon.]
)

#figure(
  grid(
    columns: 2,
    gutter: 10pt,

    [
      #image("./figures/m4_phase/u-net_bare_baseline_results.webp", width: 100%)
    ],

    [
      #image("./figures/m4_phase/2phase_results.webp", width: 100%)
    ]
  ),
  caption: [A U-net baseline (balra) modell illetve a kétfázisú modell (jobboldali) által elért eredmények a teszt adathalmazon.]
)

//todo tovabba kellenek majd abrak a tanulasi gorbekrol meg a teljesitmeny metrikakrol, felsorolas szeruen adott beallitas milyen ertekeivel sikerult elerni pl. milyen lr, dropout,..stb illetve random kep a predikciora hogyan sikerult

//todo  class imbalance pixelekrol boti bboxai alapjan ujrakalkulalni

// todo yolo bboxokrol kepet berakni

//todo adatviz terv alapjan lrean curvek, eredmenyek baseline + 2fazisura

//todo egymas mellett legyenek majd baseline vs masik

//todo legyen leirva mind2 esetben a parameter konfig konkret beallitasok kodbol amit hasznaltunk

//todo yolo parametereit is leirni

//yolonal lett tobb dataset


  = Gépi tanulás modell értékelés terv

A „YOLO és U-Net architektúra terv” fejezetben bemutatott modell kiértékelése több lépésben történik. Elsőként a modell U-Net komponensére hiperparaméter-optimalizálást (HPO) végzünk, amely során különböző paraméterkonfigurációk teljesítményét hasonlítjuk össze validációs adathalmazon.

A kiválasztás alapját elsősorban a Dice-együttható képezi, mint szegmentációs teljesítménymutató. A legjobb validációs eredményt elérő konfiguráció kerül kiválasztásra, amelyet ezt követően a teszt adathalmazon értékelünk. A kétfázisú architektúra elért eredményeit összevetjük egy U-net baseline modellel, a baseline modell beállításai, keresési tere megegyezik a kétfázisú modell U-net fejével, csak ebben az esetben mellőzzük a Yolo háló használatát.

= Alkalmazás

== Backend

A backend egy FastAPI alapú REST API, amely panoráma fogászati röntgenképek feldolgozására szolgál. A rendszer egy `/predict` endpointot biztosít, amely több képet képes egyszerre fogadni multipart/form-data formában.

A feldolgozási pipeline lépései:
- a feltöltött képek beolvasása UploadFile objektumként
- képek dekódolása PIL és NumPy segítségével RGB tömbbé
- szegmentációs modell futtatása (model.predict)
- mask és overlay generálása
- eredmény PNG-be kódolása, majd base64 formában visszaküldése

A backend CORS middleware-t használ, így a frontend közvetlenül tud kommunikálni vele localhost környezetben.

== Model

A rendszer egy U-Net alapú szegmentációs architektúrára épül, amely fogszuvasodási régiókat detektál panoráma röntgenképeken. A jelenlegi implementáció egy mock modell, amely ellipszis alakú régiókkal szimulálja a caries területeket, de a pipeline kompatibilis valódi deep learning modellekkel is.

== Frontend (Streamlit UI)

A frontend egy Streamlit alapú interaktív webalkalmazás, amely lehetővé teszi több röntgenkép feltöltését, kezelését és a szegmentációs eredmények vizualizálását.

A rendszer session state alapú, így a feltöltött képek és predikciók a felhasználói munkamenetben megmaradnak.

A frontend fő komponensei:
- file uploader (több fájl drag & drop feltöltéssel)
- "Run Segmentation" gomb a modell futtatásához
- "Remove All Images" gomb az állapot törléséhez
- képek kiválasztása és megjelenítése
- eredmények vizualizálása (original + overlay)

A frontend HTTP POST kéréssel kommunikál a backenddel, a képeket bináris formában küldi el, majd a válaszként kapott base64 encoded overlay képeket dekódolja és megjeleníti.

== Funkciók

=== Feltöltés (Drag & Drop)
A felhasználó több panoráma röntgenképet tölthet fel egyszerre drag & drop segítségével. A Streamlit file uploader kezeli a fájlok beolvasását és session state-be mentését.

=== Szegmentáció futtatása
A "Run Segmentation" gomb megnyomásával a frontend HTTP POST kérést küld a backend /predict endpointjára. A backend visszaadja a szegmentációs eredményeket, amelyeket a frontend eltárol és megjelenít.

#figure(
  image("./figures/app/upload_download.PNG", width: 80%),
  caption: [Képek feltöltése/letörlése funkció, illetve a szegmentáció futtatásához szükséges gomb],
)
=== Eredmények megjelenítése
A UI két panelen jeleníti meg az adatokat:
- bal oldalon az eredeti röntgenkép
- jobb oldalon a szegmentált overlay, amely kiemeli a detektált caries régiókat

#figure(
  image("./figures/app/result_view.PNG", width: 80%),
  caption: [Eredmények megjelenítése.],
)






=== Képek kezelése (Manage)
A felhasználó kiválaszthat egy képet egy listából, majd törölheti azt a session state-ből. A törlés frissíti a UI állapotát és a kapcsolódó predikciókat is.

#figure(
  image("./figures/app/manage_images.PNG", width: 80%),
  caption: [Kép kezelés menüpont bemutatás.],
)

=== Állapotkezelés
A rendszer Streamlit session_state-et használ a feltöltött képek, predikciók és aktuális index tárolására, így biztosítva a konzisztens felhasználói élményt a UI újrarenderelése során.

== Kommunikáció

A frontend és backend HTTP alapú kommunikációt használ:
- kérés: multipart/form-data (képek bináris formában)
- válasz: JSON (filename, model, base64 overlay)
- vizualizáció: base64 → PNG → PIL Image
=== Szegmentáció vizualizációs módok

(Binary vs Soft Heatmap)

A rendszer két különböző vizualizációs módot támogat a szegmentációs eredmények megjelenítésére, amelyek a felhasználói igények szerint választhatók.

==== Binary (threshold-alapú szegmentáció)

Ebben a módban a modell által generált valószínűségi térkép (probability map) egy előre megadott küszöbérték (threshold) alapján binarizálásra kerül.

- a kimenet csak két értéket tartalmaz: 0 és 1
- a 0 érték a háttér (nem caries)
- az 1 érték a detektált caries régió
- a vizualizáció piros színnel jelöli a pozitív pixeleket
- a döntés determinisztikus, küszöb alapú

Ez a mód diagnosztikai szempontból értelmezhetőbb, mivel egyértelmű határokat ad a detektált elváltozásokhoz.

==== Soft Heatmap (valószínűségi alapú vizualizáció)

Ebben a módban a modell kimenete nem kerül binarizálásra, hanem a teljes valószínűségi eloszlás vizuálisan kerül megjelenítésre.

- a pixelértékek 0–1 közötti valószínűségek
- ezek 0–255 skálára vannak normalizálva
- a színezés colormap (pl. JET / VIRIDIS / INFERNO) segítségével történik
- a magasabb valószínűség melegebb színeket (sárga, piros) kap
- az alacsonyabb valószínűség hidegebb színeket (kék, zöld)

A soft heatmap célja a modell bizonytalanságának és döntési gradienseknek a vizualizációja.

#figure(
  image("./figures/app/soft_heatmap.PNG", width: 80%),
  caption: [heatmap funkció bemutatás.],
)


==== Overlay generálás

Mindkét esetben a végső megjelenítés az eredeti röntgenkép és a szegmentációs eredmény kombinálásával történik.

- binary mód esetén a maszk piros overlayként jelenik meg
- soft mód esetén a heatmap félig áttetszően kerül rá az eredeti képre

A keverés OpenCV `addWeighted` függvénnyel történik, amely biztosítja az anatómiai struktúrák és a predikciók egyidejű láthatóságát.

A vizualizációs mód a felhasználó által dinamikusan választható, és minden kérés során a backend újragenerálja az overlay képet a kiválasztott módnak megfelelően.


  = SOTA MODELLEK

  == CariesNet

A CariesNet egy mélytanulás-alapú szegmentációs modell, amely több stádiumú szuvas léziók detektálására készült panorámaröntgen-felvételeken. Architektúrája a U-Net struktúrájára épül, amelyet egy teljes skálájú axiális figyelmi (Full-Scale Axial Attention – FSAA) modullal, valamint egy részleges enkóder modullal egészítettek ki a szegmentációs teljesítmény javítása érdekében, különös tekintettel a kisebb kiterjedésű léziókra. A modellt 1159 panorámafelvételből álló adathalmazon tanították, amely összesen 3217 annotált szuvas régiót tartalmazott (kezdeti, középsúlyos és mély szuvasodás). A CariesNet 93,64%-os átlagos Dice-együtthatót és 93,61%-os pontosságot ért el, ezzel felülmúlva az olyan alapmodelleket, mint a U-Net, a DeepLabV3+ és a PraNet. Az FSAA modul különösen a lézióhatárok pontosabb kirajzolását segíti elő, míg a részleges enkóder a magas szintű jellemzők aggregálásával járul hozzá a precízebb szegmentációhoz #cite(<zhu2023cariesnet>).
== CariesSeg

A CariSeg négy neurális hálózat integrációján alapul, és 99,42%-os pontosságot (accuracy) ért el a fogszuvasodás detektálásában panorámaröntgen-felvételeken. A rendszer első komponense egy U-Net architektúrán alapuló modell, amely a fogak régióját szegmentálja, majd a felvételt az érdeklődési területre fókuszálva kivágja. A második komponens a szuvas léziók szegmentálását végzi egy három architektúrából (U-Net, Feature Pyramid Network és DeepLabV3+) álló ensemble modell segítségével. A fogazonosításhoz két egyesített adathalmazt használtak: a Tufts Dental Database 1000 panorámaröntgen-felvételét, valamint egy további, 116 anonim panorámafelvételből álló adatbázist, amely a Noor Medical Imaging Centerben (Qom) készült. A szuvasodás szegmentációhoz 150 panorámaröntgen-felvételt tartalmazó adatbázist alkalmaztak, amely az Iuliu Hațieganu Orvosi és Gyógyszerészeti Egyetem Száj- és Állcsontsebészeti, valamint Radiológiai Tanszékéről származik. Az ensemble megközelítés az egyes modellek komplementer erősségeit egyesíti, ezáltal kiemelkedő szegmentációs teljesítményt elérve #cite(<muarginean2024teeth>).

== End‑to‑end mélytanulás alapú rendszer fogszuvasodás szegmentációra
 Ebben a munkában egy U-Net-alapú architektúrát alkalmaztak a léziók pixelenkénti szegmentálására, valamint egy ResNet-50 osztályozó hálózatot a többosztályos besorolásra (nincs szuvasodás, zománc szuvasodás, dentin szuvasodás). A U-Net modell 0,89-es Dice-együtthatót ért el, ami a lézióhatárok precíz meghatározását jelzi. A ResNet-50 osztályozó 93,2%-os összpontosságot mutatott, az egyes kategóriák szerinti pontosság pedig 95% (nincs szuvasodás), 91,1% (zománc szuvasodás) és 90,4% (dentin szuvasodás) volt #cite(<marwaha2025end>).

= MLOps Platform

A projekt során az MLOps (Machine Learning Operations) folyamatok menedzselésére a felhőalapú Weights & Biases (W&B) #cite(<wandb>) platformot választottuk. A W&B segítségével szisztematikusan követhetjük az egyes tanítási kísérleteket, verziózhatjuk a modelljeinket, és biztosíthatjuk az eredmények reprodukálhatóságát, mindezt lokális infrastruktúra-karbantartás nélkül.

A W&B integrációjával a Python kódba (a `wandb` könyvtáron keresztül) a folyamatokat az alábbi három fő pillérre alapoztuk:

1. W&B Tracking (Kísérletkövetés): Ezzel naplózzuk az összes kísérletet, beleértve a hiperparamétereket (például tanulási ráta, kötegméret), a kiértékelési metrikákat (Dice-együttható, betanítási és validációs veszteség) és a tanítás során generált vizualizációkat. Ez kritikus fontosságú a U-Net és Large U-Net variánsok szisztematikus összehasonlításakor.

2. W&B Artifacts: Itt tároljuk a betanított modellek súlyait. Az Artifacts rendszer segítségével könnyedén kezelhetjük a különböző adathalmaz-verziókat és a felhőbe mentett modellsúlyokat (checkpoints), elősegítve a reprodukálhatóságot és az elosztott tesztelést.

3. Erőforrás Monitoring (System Metrics): A W&B transzparens módon, valós időben rögzíti a hardver-erőforrások állapotát a betanítás során (GPU memóriahasználat, hőmérséklet, feldolgozóegység teljesítménye), amely elengedhetetlen a dedikált hardver – jelen esetben a helyi NVIDIA RTX GPU – hatékony kihasználásához.

= Eredmények


= Konklúzió

#bibliography("ref.bib", style: "ieee", title: auto)
