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
- Modellnek egy a U-net hálózatnak egy továbbfejlesztett verzióját használjuk
- Baseline modellnek  sima U-net hálót használnánk
- Adathalmaznak a DC1000 adathalmazt használjuk
- Kiértékelési metrikáknak a Precision, Recall, F1 illetve IoU metrikákat használjuk
//- Cél: legalább egy 0.65 Precision illetve Recall elérése 

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
- Augmentálás: 
  + Véletlenszerű vízszintes tükrözés
  + Eltolás, skálázás és forgatás
  + Véletlenszerű fényerő- és kontrasztállítás
- Adathalmaz felosztása tanító, teszt és validációs adathalmazra



= Mélytanulási architktúrák

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
