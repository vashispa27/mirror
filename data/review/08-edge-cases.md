# 08-edge-cases

6 conversations · 100 messages · 33 questions

## 08-edge-cases-1 — landlord-repair-thread (2026-02-05)

` 1` **Aditi:** Bathroom ki deewar pe jo seepage tha, wo phir se aa gaya. Kal subah se paani tapak raha hai.
` 2` **Rohan:** Wahi wala issue jo pichhle mahine fix hua tha? Ya koi naya spot hai? Photo bhej sakti ho?
` 3` **Aditi:** Wahi hai, usne sirf upar se paint kiya tha. Asli problem andar hai, isliye wo laut aaya.
` 4` **Rohan:** Mishra ji ko likh ke bhejo, likhne me proof rehta hai. Phone pe baat karne se baad me mukar jaate hain.
` 5` **Aditi:** Maine usko message bhej diya tha, dekha bhi nahi. Uski galti hai, kaam adhoora chhod diya.
` 6` **Rohan:** Do din ka time do, phir likh ke bolo ki naye plumber ka kharcha wo dega. Warna deposit se kaat lena.
` 7` **Aditi:** Wahan pe rehna mushkil ho gaya hai. Us cheez ki smell se sar dard karta hai, aur bacche bhi pareshan hain.
` 8` **Rohan:** Sach me stressful hai. Ek temporary fix karwa lo, tape aur sealant se, aur usko bill bhej do.
` 9` **Aditi:** Aaj Kaleem aayega, wo pehle bhi kaam karta tha. Usko wahi jagah dikhaungi jo seepage wali hai.
`10` **Rohan:** Achha, usko bolo ki andar ka plaster khol ke dekhe. Sirf bahar se seal karna kaam nahi karega.
`11` **Aditi:** Usne kaha ki teen hazaar lagega agar plaster kholna pada. Main soch rahi hoon ki ye kharcha kispe aayega.
`12` **Rohan:** Mishra ji pe. Unhone kharab fitting lagayi thi, jo teen mahine me hi leak ho gayi. Likho ki ye unki zimmedari hai.
`13` **Aditi:** Theek hai, main usko ye sab likh ke bhejti hoon. Wo mana kar sakta hai par likha hua kaam aayega.
`14` **Rohan:** Bilkul. Agar wo phir bhi na maane, to society ke group me daal do. Wahan sab dekh lenge.
`15` **Aditi:** Haan, aur main us cheez ki photo bhi rakh rahi hoon. Sab evidence save kar rahi hoon.
`16` **Rohan:** Sahi. Ab chain se so jao, subah Kaleem ko call kar lena.

### Questions

- `exact_phrase` Mishra ji pe kharcha daalne ke liye kya likhna hai?
  - evidence: `Likho ki ye unki zimmedari hai.`
- `semantic` Plumber kaun hai jo dubara aa raha hai?
  - evidence: `Aaj Kaleem aayega, wo pehle bhi kaam karta tha`
- `multihop` Teen hazaar ka kharcha kaun dega?
  - evidence: `Mishra ji pe.`
- `attribution` Teen hazaar ka estimate kisne diya tha?
  - evidence: `Usne kaha ki teen hazaar lagega`
- `situation` Mishra ji phone pe baat karke mukar jaate hain, to kya karna chahiye?
  - evidence: `likhne me proof rehta hai`

## 08-edge-cases-2 — family-doctor-advice (2026-02-18)

` 1` **Aditi:** Vikas ne bola ki wo humare saath Goa nahi ja raha. Uska boss leave nahi de raha.
` 2` **Rohan:** Achha, to wo alag plan bana raha hai? Usko manane ki koshish karogi ya plan change karenge?
` 3` **Aditi:** Uski marzi hai. Doctor Nair ne bhi usko travel se mana kiya tha, BP high hai uski.
` 4` **Rohan:** Ye important hai. Agar doctor ne mana kiya hai to usko rest karna chahiye, trip baad me ho sakti hai.
` 5` **Aditi:** Par Sunita ma'am bhi keh rahi hain ki office me sab log chhutti le rahe hain, to usko guilt feel hota hai.
` 6` **Rohan:** Office wale guilt ka pressure trip se zyada important nahi hai. Health pehle.
` 7` **Aditi:** Haan. Aur landlord Batra uncle ne kaha ki building ka painting ka kaam usi hafte hoga, to ghar me shor rahega.
` 8` **Rohan:** To ghar me rehna bhi mushkil hai. Kya Vikas kisi cousin ke ghar ruk sakta hai us hafte?
` 9` **Aditi:** Wo keh raha hai ki usko akela rehna hai. Teacher wali behen, Meera, ne bhi kaha ki wo thoda low feel kar raha hai.
`10` **Rohan:** Meera ka observation sahi lagta hai. Usko call karke baat karo, zabardasti mat karo.
`11` **Aditi:** Main usko kal call karungi. Doctor Nair se report bhi mangwa rahi hoon.
`12` **Rohan:** Report aane pe decide karna. Abhi koi plan fix mat karo.
`13` **Aditi:** Theek hai. Sunita ma'am ko main message karti hoon ki wo leave de de.
`14` **Rohan:** Haan, par Vikas ki marzi poocho pehle. Uski life hai, uska decision hona chahiye.
`15` **Aditi:** Sahi kaha. Main usko batati hoon ki hum log samajhte hain.
`16` **Rohan:** Perfect. Chalo, tension kam karo. Sab theek ho jayega.

### Questions

- `attribution` Doctor ne travel ke baare me kya kaha tha?
  - evidence: `Doctor Nair ne bhi usko travel se mana kiya tha`
- `attribution` Painting ka kaam kisne bataya tha?
  - evidence: `landlord Batra uncle ne kaha ki building ka painting ka kaam usi hafte hoga`
- `exact_phrase` Meera ne kya observation di thi?
  - evidence: `Meera, ne bhi kaha ki wo thoda low feel kar raha hai.`
- `semantic` Bhai ko aaram kyun zaroori hai?
  - evidence: `usko rest karna chahiye`
- `situation` Painting wale hafte Vikas kahan ruk sakta hai?
  - evidence: `kisi cousin ke ghar ruk sakta hai`
- `unanswerable` Doctor Nair ka clinic kahan hai?
  - evidence: (none — unanswerable)

## 08-edge-cases-3 — parcel-tracking-details (2026-03-09)

` 1` **Aditi:** Mera parcel order SWX-4471-QP kahan hai? Tracking me teen din se same status hai.
` 2` **Rohan:** Us number ko courier wali site pe daalo, wahan se live location milegi. Kya customer care ko call kiya?
` 3` **Aditi:** Haan, unka number 90000 12345 hai, par wo automated message pe le jaate hain.
` 4` **Rohan:** Twitter pe tag karo, wahan insaan reply karta hai. Order id likh do saath me.
` 5` **Aditi:** Achha. Waise delivery address galat chhap gaya tha, Flat 3B, Neel Kamal Apartments, Kalyani Nagar.
` 6` **Rohan:** Agar address galat hai to parcel wapas jayega. Courier office jaake address correction form bharo.
` 7` **Aditi:** Maine online form bhar diya tha, par unhone bola ki Rs 2,499 ka COD confirm karna padega.
` 8` **Rohan:** COD hai? To cash ready rakho. UPI wale driver bhi hote hain, par cash safe hai.
` 9` **Aditi:** Haan, SWX-4471-QP 12 March ko aayega. Us din main ghar pe nahi hoon.
`10` **Rohan:** Kisi ko ghar pe rakho, warna driver wapas chala jayega. Ya courier office se pickup kar lo.
`11` **Aditi:** Mera bhai 12 March ko free hai, wo le lega. Usko OTP bhejna padega, order wale number pe aayega.
`12` **Rohan:** Haan, aur driver ko bhai ka naam bata dena. Gate pe koi dikkat na ho.
`13` **Aditi:** Haan wahi hai. Waise ek aur parcel aana hai, order GH-8802-LM, usme sirf ek book hai.
`14` **Rohan:** Wo alag shipment hai, uska tracking alag hoga. Dono ke liye alag alerts set karo.
`15` **Aditi:** Theek hai. Book wala 18 March ko aa raha hai, us din main office me hoon.
`16` **Rohan:** Book ka pickup bhi bhai se karwa lo. Ek hi trip me dono nikal jayenge.

### Questions

- `exact_phrase` SWX-4471-QP wala parcel kis din aayega?
  - evidence: `SWX-4471-QP 12 March ko aayega`
- `paraphrase` Customer care ka number kya hai?
  - evidence: `unka number 90000 12345 hai`
- `semantic` Pata bigad gaya ho to seedha kahan jana hai?
  - evidence: `Courier office jaake address correction form bharo`
- `temporal` Book wala parcel kab aa raha hai?
  - evidence: `Book wala 18 March ko aa raha hai`
- `multihop` COD me kitna paisa aur kis din dena hai?
  - evidence: `Rs 2,499 ka COD confirm karna padega`

## 08-edge-cases-4 — two-bike-service-visits (2026-03-24)

` 1` **Aditi:** Bike phir se awaaz kar rahi hai. Pichhli baar chain ka problem tha, is baar kuch aur lagta hai.
` 2` **Rohan:** Pehli service kis gaadi wale ne ki thi? Ravi Auto Works wala kaam theek tha ya nahi?
` 3` **Aditi:** Pehli baar January me Ravi Auto Works gaya tha, 1,150 rupaye liye the. Do hafte tak theek chali.
` 4` **Rohan:** Do hafte me hi problem? To usne sirf chain tight ki hogi, adjust nahi kiya.
` 5` **Aditi:** Ho sakta hai. Ab is baar March me Speed Motors ja rahi hoon, brake pads change karwane hain.
` 6` **Rohan:** Brake pads ka kharcha alag hota hai. Kya unhone estimate diya?
` 7` **Aditi:** Haan, 2,300 rupaye bola. Front disc bhi thoda ghis gaya hai.
` 8` **Rohan:** Disc ka kaam mehnga padta hai. Do jagah se quote lo, warna seedha paisa lag jayega.
` 9` **Aditi:** Speed Motors wala bhaiya kehta hai ki disc abhi theek hai, sirf pads badal do.
`10` **Rohan:** To sirf pads karwao. Disc baad me, jab zyada problem ho.
`11` **Aditi:** Haan. Waise pehli baar jo bill mila tha, wo maine kahin rakh diya tha.
`12` **Rohan:** Bill mil gaya to warranty claim ho sakta hai. Dhoondh lo, chain wala kaam usi ka tha.
`13` **Aditi:** Nahi mil raha. Lagta hai ghar shift karte waqt kho gaya.
`14` **Rohan:** Koi baat nahi. Ab Speed Motors ka bill sambhal ke rakho, wahi kaam aayega.
`15` **Aditi:** Theek hai. Aur ek baat, pehli baar mera phone bike pe hi reh gaya tha, wahan se wapas mila.
`16` **Rohan:** Achha, wo Ravi wala garage imaandar lagta hai. Is baar bhi check karke jao.

### Questions

- `multihop` Pehli service ka kharcha aur garage kya tha?
  - evidence: `Ravi Auto Works gaya tha, 1,150 rupaye liye the`
- `exact_phrase` Speed Motors wali service me kya change karwana hai?
  - evidence: `Speed Motors ja rahi hoon, brake pads change karwane hain`
- `semantic` Pehla mechanic ne kaam adhoora chhoda tha kya?
  - evidence: `usne sirf chain tight ki hogi`
- `temporal` Pehli service kis mahine me hui thi?
  - evidence: `Pehli baar January me`
- `unanswerable` Speed Motors wale bhaiya ka poora naam kya hai?
  - evidence: (none — unanswerable)

## 08-edge-cases-5 — job-stress-and-shift-swap (2026-04-11)

` 1` **Aditi:** Aaj office me bahut bura din tha. Meeting me sabke saamne mujhe daant padi.
` 2` **Rohan:** Yaar, ye sunke bura laga. Kya hua tha exactly?
` 3` **Aditi:** Mera project late ho gaya, aur Sunita ma'am ne bola ki main team ko update nahi de rahi.
` 4` **Rohan:** Par tum to roz update bhejti ho. Ye unfair lagta hai, tumhari mehnat koi nahi dekh raha.
` 5` **Aditi:** Bas wahi. Ghar aake roya bhi. Ab lagta hai ye job hi galat hai.
` 6` **Rohan:** Ek bura din poora career decide nahi karta. Aaj rest karo, kal soch samajh ke baat karenge.
` 7` **Aditi:** Aur ek problem hai. Kal night shift hai, par mere papa ki tabiyat kharab hai, unko doctor ke paas le jana hai.
` 8` **Rohan:** To shift swap karo kisi colleague se. Kis se swap ho sakta hai?
` 9` **Aditi:** Nikhil se ho sakta hai, wo raat wali shift pasand karta hai. Usko message kiya hai.
`10` **Rohan:** Achha. Aur papa ko subah le jao, doctor 10 baje se pehle free rehta hai.
`11` **Aditi:** Haan, appointment 10:30 baje ka hai. BP ki dawa bhi khatam ho gayi hai, refill karwana hai.
`12` **Rohan:** Dawa ki list bhejo, main pharmacy se le aaunga. Tum office sambhalo.
`13` **Aditi:** Thank you. Par mujhe lagta hai main sab kuch akeli handle kar rahi hoon, koi support nahi.
`14` **Rohan:** Main hoon na. Sunita ma'am ki baat kal bhool jaogi, par health wali cheez wait nahi karti.
`15` **Aditi:** Sahi. Nikhil ne haan bol diya, shift swap ho gaya. Ab kal raat free hoon.
`16` **Rohan:** Badhiya. Papa ko le jao, aur shaam ko apne liye kuch achha khana mangwa lo.
`17` **Aditi:** Haan. Aur kal Sunita ma'am se baat karke project update wala issue clear karungi.
`18` **Rohan:** Haan, shaant dimaag se. Aaj tumne bahut jhela hai, ab aaram karo.

### Questions

- `attribution` Meeting me kisne bola tha ki main update nahi de rahi?
  - evidence: `Sunita ma'am ne bola ki main team ko update nahi de rahi`
- `multihop` Papa ko doctor ke paas le jaana hai, isliye shift swap kis se hua?
  - evidence: `Nikhil se ho sakta hai, wo raat wali shift pasand karta hai.`
- `temporal` Papa ka appointment kis samay ka hai?
  - evidence: `appointment 10:30 baje ka hai`
- `exact_phrase` Nikhil ne shift swap pe haan bola tha?
  - evidence: `Nikhil ne haan bol diya`
- `semantic` Kya ye sab temporary hai? Mujhe break lena chahiye kya?
  - evidence: `Aaj rest karo`
- `attribution` Dawa laane ki zimmedari kisne li thi?
  - evidence: `main pharmacy se le aaunga`

## 08-edge-cases-6 — wedding-gift-coordination (2026-04-27)

` 1` **Aditi:** Nisha ki shaadi hai 15 May ko. Gift ke liye group bana hai, par sab log confused hain.
` 2` **Rohan:** Kya gift decide hua ya abhi discussion me hai? Budget bhi batao.
` 3` **Aditi:** Kavita ne bola ki hum sab milke ek washing machine dein. Usne price bhi nikaal liya.
` 4` **Rohan:** Washing machine bada gift hai, ghar me jagah bhi chahiye. Nisha ke ghar me space hai kya?
` 5` **Aditi:** Wo nayi flat me shift ho rahi hai, wahan sab kuch chahiye. Par Vikas kehta hai cash dena better hai.
` 6` **Rohan:** Dono ke point valid hain. Cash safe hai, par machine zyada yaad rehta hai.
` 7` **Aditi:** Aur ek baat, Kavita ne kaha ki wo 8,000 rupaye de sakti hai, baaki hum baant lenge.
` 8` **Rohan:** Total kitna ho raha hai? Machine ka price aur collection ka hisaab banao.
` 9` **Aditi:** Machine 32,000 ki hai. Aath log hain, to har ek pe 4,000 aata hai.
`10` **Rohan:** Theek hai. Par Kavita ka 8,000 alag hai, to usko count karke baaki baant do.
`11` **Aditi:** Haan. Wo bol rahi hai ki wo delivery bhi arrange kar degi, uske bhai ka transport hai.
`12` **Rohan:** Achha, to delivery ka tension nahi. Bas payment ka hisaab group me daal do.
`13` **Aditi:** Vikas phir bhi cash pe adamant hai. Wo kehta hai ki machine duplicate ho sakti hai.
`14` **Rohan:** Uska matlab Nisha ke paas already machine ho sakti hai. Kavita se confirm karwao.
`15` **Aditi:** Sahi, main Kavita se poochhungi. Waise gift pe koi deadline hai kya shaadi se pehle?
`16` **Rohan:** Gift 15 May se pehle pahunchana chahiye. Order 10 May tak de dena.
`17` **Aditi:** Theek hai. Vikas ko bhi manana padega, warna wo alag gift le aayega.
`18` **Rohan:** Usko group me daalo, sabke saamne baat hui to wo zyada nahi bolega.

### Questions

- `attribution` Cash dene ki salah kisne di thi?
  - evidence: `Par Vikas kehta hai cash dena better hai`
- `exact_phrase` Kavita kitne paise de rahi hai?
  - evidence: `Kavita ne kaha ki wo 8,000 rupaye de sakti hai`
- `multihop` Machine ka total price aur per head hissa kya hai?
  - evidence: `Machine 32,000 ki hai.`
- `semantic` Shaadi ki tarikh se kitne din door saman mangwana hai?
  - evidence: `Order 10 May tak de dena`
- `situation` Vikas machine ke against kyun hai?
  - evidence: `Wo kehta hai ki machine duplicate ho sakti hai.`
- `unanswerable` Nisha ke flat ka number kya hai?
  - evidence: (none — unanswerable)
