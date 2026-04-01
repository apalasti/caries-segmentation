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

1. *Detektor alapmodell (YOLO):* fog-régiók detektálása bounding box szinten.
2. *Szegmentáló alapmodell (U-Net):* pixel-szintű szuvasodás maszkok tanítása.
3. *Konjunkciós modell (YOLO + U-Net):* a YOLO által detektált régiók kivágása, majd lokális maszkfinomítás U-Net segítségével.
4. *Összehasonlítás és ablatív vizsgálat:* külön értékeljük a YOLO-only, U-Net-only és a konjunkciós pipeline teljesítményét.

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
== Adatelőkészítési terv
Az alábbiakban felsoroljuk azokat a lépéseket, amelyeket az adatok megfelelő előkészítése érdekében szükséges elvégezni:
- Képek átméretezése
- Normalizálás
- Augmentálás (online, részletek: *Online Augmentáció* fejezet): 
  + Véletlenszerű vízszintes tükrözés
  + Véletlenszerű méretarány (`RandomScale`; eltolás és forgatás nélkül)
  + Opcionális elastic deformáció (projekt `config.toml` szerint jelenleg kikapcsolva)
  + Véletlenszerű fényerő- és kontrasztállítás
- Adathalmaz felosztása tanító, teszt és validációs adathalmazra

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

= Adatvizualizációs terv

- *Adathalmaz bemutatása*
  - *Cél:* bemutatni, hogy milyen képeken tanul a modell
  - *Vizualizációk:*
    - panoráma röntgen képek mintái: képrács (image grid)
    - adathalmaz felosztása tanító, validációs és teszt adathalmazra: pie chart
    - szuvas / nem szuvas pixelek aránya: pie chart


- *Annotáció vizualizáció*
  - *Cél:* megmutatni, hogy mit tanul a modell
  - *Vizualizációk:*
    - eredeti röntgen kép: image
    - ground truth maszk: maszk kép
    - overlay (maszk rárajzolva az eredeti képre): overlay image
    - több példa összehasonlítása: képrács (original – mask – overlay)
/*
- *Pixeleloszlás vizsgálata*
  - *Cél:* az osztályok közötti egyensúlytalanság vizsgálata
  - *Vizualizációk:*
    - szuvas pixelek száma képenként: histogram
    - szuvas / nem szuvas pixel arány képenként: boxplot
    - szuvas területek gyakori elhelyezkedése a képeken: heatmap
*/
/*- *Adat augmentáció vizualizáció*
  - *Cél:* bemutatni az alkalmazott adatnövelési módszereket
  - *Vizualizációk:*
    - eredeti kép és augmentált változatok: képrács
      - rotate: transform példa
      - flip: transform példa
      - contrast változtatás: intensity transform
*/
- *Modell predikció vizualizáció*
  - *Cél:* a modell szegmentációs eredményeinek bemutatása
  - *Vizualizációk:*
    - eredeti kép: image
    - ground truth maszk: mask
    - modell predikció: predicted mask
    - különbség (ground truth vs prediction): difference map
      - false positive pixelek: piros jelölés
      - false negative pixelek: kék jelölés

- *Tanulási görbék*
  - *Cél:* a modell tanulási folyamatának vizsgálata
  - *Vizualizációk:*
    - training loss alakulása epoch szerint: vonaldiagram
    - validation loss alakulása epoch szerint: vonaldiagram
    - Dice score alakulása epoch szerint: vonaldiagram
    - IoU alakulása epoch szerint: vonaldiagram

- *Teljesítmény metrikák*
  - *Cél:* a modell végső teljesítményének értékelése
  - *Vizualizációk:*
    - confusion matrix pixel szinten: confusion matrix diagram
    - Dice score értékek: oszlopdiagram
    - IoU értékek: oszlopdiagram
    - precision és recall értékek: oszlopdiagram

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

== Pixeleloszlás vizsgálata

Az egyes osztályok pixeleloszlását az alábbi ábrák mutatják a különböző adathalmaz részekben. Megfigyelhető, hogy az osztályok aránya nem teljesen kiegyensúlyozott, ami hatással lehet a modell tanulására.


#figure(
  grid(
    columns: 3,
    gutter: 10pt,

    [
      #image("./figures/dataset_introduction/class_ratio_train.png", width: 100%)
    ],

    [
      #image("./figures/dataset_introduction/class_ratio_val.png", width: 100%)
    ],

    [
      #image("./figures/dataset_introduction/class_ratio_test.png", width: 100%)
    ],
  ),
  caption: [Az osztályeloszlás összehasonlítása a tanító, validációs és teszt halmazokban.]
)


/*== Adat augmentáció vizualizáció ezt elhagyjuk mivel ugyis random*/


== Modell predikció vizualizáció

#figure(
  image("./figures/evaluation/confusion_matrix.png", width: 100%),
  caption: [Konfúziós mátrix.]
)

A tanító, validációs és teszt halmazokra vonatkozó osztályeloszlás ábrák bemutatják, hogy a szuvas és nem szuvas pixelek aránya nem teljesen kiegyensúlyozott.
#figure(
  image("./figures/evaluation/sample_1.png", width: 100%),
  caption: [Egy példa a modell predikciójára.]
)
== Tanulási görbék

A tanulási görbék ábrázolják a modell veszteség- és pontossági metrikáinak változását az epoch-ok  során  tanító illetve  validációs adathalmazon.
#figure(
  image("./figures/training/training_curves.png", width: 100%),
  caption: [Tanulási görbék]
)
== Teljesítmény metrikák

A konfúziós mátrix és a különböző teljesítménymutatók oszlopdiagramja összefoglaló képet ad a modell végső pontosságáról, a false positive és false negative hibákról, valamint a Dice, IoU, precision és recall értékekről.
#figure(
  image("./figures/evaluation/metrics_barplot.png", width: 100%),
  caption: [Teljesítménymutatók oszlopdiagramon.]
)



= Mélytanulási architektúrák

== Detekció és Osztályozás

A detekció olyan képfeldolgozási feladat, amelyben az objektumok helyét határoló dobozokkal jelöljük ki. Az osztályozás célja, hogy a képeket vagy képrészleteket a megfelelő osztály(ok)ba sorolja.

  === YOLO és konjunkciós architektúra

Jelen munkában a korábbi ResNet-alapú megközelítés helyett YOLO-stílusú detektort alkalmazunk a fog-régiók gyors lokalizálására, majd a detektált régiókat U-Net alapú maszkolással finomítjuk. A detektor implementációja kompakt, egyléptékű, YOLOv5-szerű felépítést követ.

A YOLO detektor fő komponensei:
- *Backbone:* egymásra épülő konvolúciós blokkok (`Conv2d + BatchNorm2d + SiLU`), amelyek többlépcsős leskálázással robusztus jellemzőtérképet építenek.
- *Detekciós fej (detection head):* $1 times 1$ konvolúció, amely horgonypontonként (anchor) a következőket becsli: dobozparaméterek $(x, y, w, h)$, objektumosság és osztályvalószínűség.
- *Dekódolás és szűrés:* sigmoid aktiváció, rácskoordináta-alapú visszaskálázás, majd Non-Maximum Suppression (NMS) a duplikált dobozok eltávolítására.

A detektor kimeneti csatornaszáma a következő:

$ C_"out" = A * (5 + C) $

ahol $A$ az anchorok száma, $C$ pedig az osztályok száma. A képletben szereplő $5$ a YOLO doboz-leírás fix komponenseit jelenti: $(x, y, w, h)$ koordinátaparaméterek + objektumossági pontszám (objectness) @redmon2016yolo.

A jelenlegi implementációban nem használunk közös, end-to-end kombinált
veszteségfüggvényt a YOLO és az U-Net között. A két modell külön lépésben tanul:
először a YOLO detektor, majd külön az U-Net a YOLO által kijelölt régiókon.

Az új architekturális elem a *YOLO + U-Net konjunkciós blokk*:
- a YOLO által detektált bounding box régiókat kivágjuk,
- opcionális paddinget adunk a kontextus megőrzésére,
- a kivágást fix U-Net bemeneti méretre mintavételezzük,
- az U-Net lokális szegmentációt készít,
- a bináris maszkot visszavetítjük az eredeti képre és régiónként egyesítjük.

Ezzel a felépítéssel a YOLO biztosítja a gyors régió-jelölést, míg az U-Net a pixelek szintjén pontosítja a szuvas területek határát.

==== YOLOv5 fő metódusai

Az implementáció működését az alábbi fő metódusok írják le @yolov5:

- *Feature-extrakció és előrecsatolás (forward):* a backbone jellemzőtérképeket állít elő, majd a detekciós fej anchoronként becsli a dobozparamétereket, objektumosságot és osztálypontszámokat.
- *Predikció dekódolás:* a nyers kimenetekből (logitokból) rács- és anchor-alapú transzformációval képi koordinátákra visszavetített dobozok készülnek.
- *Küszöbölés és NMS:* a gyenge találatok szűrése után Non-Maximum Suppression eltávolítja az átfedő, redundáns dobozokat.
- *Tanítási célfüggvények (detektor szinten):* külön komponensek kezelik a dobozregressziót, az objektumosságot és az osztályozást.


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
  === U-Net

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

Jelen kutatásban az eredeti U-Net modellt vettük alapul, amelyet egy mélyebb, *Large U-Net* architektúrára is kiterjesztettünk, hogy a rendelkezésre álló GPU (NVIDIA RTX) memóriáját hatékonyabban használjuk ki, illetve robusztusabb reprezentációt tanulhassunk a röntgenfelvételekből. Az implementáció a Python programozási nyelven, a PyTorch keretrendszer #cite(<paszke2019pytorch>) segítségével készült.

==== Konvolúciós Blokk, Kernel, Padding és Stride

Az architektúra alapköve a duplázott konvolúciós blokk (`DoubleConv`), amely egyaránt alkalmazásra kerül az "encoder" és a "decoder" ágban. Ez a blokk két egymást követő kétdimenziós konvolúciós rétegből (`Conv2d`) áll. A konvolúciós szűrők (kernel) mérete $3 times 3$, amely standard értékként elegendő a lokális térbeli mintázatok és élek felismeréséhez.

Annak érdekében, hogy a transzformáció során a hálózat feleslegesen ne csökkentse az aktivációs térképek térbeli felbontását (magasságát és szélességét), egységnyi kitöltést (`padding=1`) alkalmaztunk. A lépésköz (`stride`) értéke a konvolúciós blokk belsejében $1$, így a konvolúciós ablak minden egyes pixelre finoman rácsúszik, megőrizve a rácsfelbontást.

A kódoló szakaszban a térbeli redukciót mindig hálózaton kívüli, exkluzív $2 times 2$-es Max Pooling réteg végzi (ahol a lépésköz is 2), ami rendre megfelezi a felépített reprezentációk felbontását. A dekódoló ág feladata ennek ellentéte, a térbeli dimenziók visszaállítása transzponált konvolúciók (`ConvTranspose2d`) segítségével ($2 times 2$-es kernel, $2$-es stride kíséretében).

==== Batch Normalization

Minden konvolúciós műveletet a PyTorch `BatchNorm2d` rétege követi a nem-lineáris aktivációs függvény (ReLU - Rectified Linear Unit) előtt. A Batch Normalizáció #cite(<ioffe2015batch>) megkönnyíti és felgyorsítja a mély neurális hálózatok betanítását azzal, hogy az egyes rétegek bemeneteinek eloszlását fixálja, redukálva az ún. belső kovariancia eltolódás (internal covariate shift) jelenségét.

Matematikailag a normálás a következőképpen történik az adott mini-batch-en belül minden csatornára függetlenül:

$ hat(x)_i = (x_i - mu_"batch") / sqrt(sigma_"batch"^2 + epsilon) $
$ y_i = gamma hat(x)_i + beta $

ahol $mu_"batch"$ a mini-batch adott térképre vonatkozó empirikus átlaga, $sigma_"batch"^2$ a varianciája, a $gamma$ (skála) és $beta$ (eltolás) pedig a hálózat által tanult paraméterek. Ennek következtében a PyTorch stabilabb gradiensáramlást tud produkálni végig az egész U-Net testen keresztül.

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

1. **W&B Tracking (Kísérletkövetés)**: Ezzel naplózzuk az összes kísérletet, beleértve a hiperparamétereket (például tanulási ráta, kötegméret), a kiértékelési metrikákat (Dice-együttható, betanítási és validációs veszteség) és a tanítás során generált vizualizációkat. Ez kritikus fontosságú a U-Net és Large U-Net variánsok szisztematikus összehasonlításakor.

2. **W&B Artifacts**: Itt tároljuk a betanított modellek súlyait. Az Artifacts rendszer segítségével könnyedén kezelhetjük a különböző adathalmaz-verziókat és a felhőbe mentett modellsúlyokat (checkpoints), elősegítve a reprodukálhatóságot és az elosztott tesztelést.

3. **Erőforrás Monitoring (System Metrics)**: A W&B transzparens módon, valós időben rögzíti a hardver-erőforrások állapotát a betanítás során (GPU memóriahasználat, hőmérséklet, feldolgozóegység teljesítménye), amely elengedhetetlen a dedikált hardver – jelen esetben a helyi NVIDIA RTX GPU – hatékony kihasználásához.

= Eredmények


= Konklúzió

#bibliography("ref.bib", style: "ieee", title: auto)
