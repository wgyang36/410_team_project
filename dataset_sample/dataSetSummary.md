# Amazon Review DataSet Format Summary

## How to Unify

| UCSD-Format    | UIUC-Format           |
| -------------- | --------------------- |
| reviewerID     | Reviews-ReviewID      |
| asin           | ProductInfo-ProductID |
| reviewerName   | Reviews-Author        |
| helpful        | `null-missing`        |
| reviewText     | Reviews-Content       |
| overall        | Reviews-Overall       |
| summary        | Reviews-Title         |
| unixReviewTime | `null-missing`        |
| reviewTime     | Reviews-Date          |

## UIUC-DataSet 

### Format-Summary

The attribute of Product Info coulde be null.

We should mainly use the `reviews part`.

For each review, it offers:

- Title
- Author
- ReviewID
- Overall Rating
- Review Content
- Review Date

All attribute are very helpful for the subsequent analysis.

For the `ProductInfo ` part, 



```
{  
   "Reviews":[  
      {  
         "Title":"Better than the park map, but do not trust it for kayaking",
         "Author":"Nicole M",
         "ReviewID":"R2JCHODG7WMHTA",
         "Overall":"3.0",
         "Content":"It is better than the NPS park map but when it comes to navigating a place like Nine Mile Pond, trust your GPS and carry a compass. It has the numbers totally inaccurate and that could cause some serious problems. They claim it is up to date, but it is not! The thing that annoys me about the folks who publish this map and field guides is they get it vetted by some knuckle head who poses as an authority but is clueless. It never occurs to them to try to use their product or watch their target market use them. So, the product is mediocre and they could care less. At one time National Geographic stood for some level of integrity but that seems to be a thing of the past.",
         "Date":"December 29, 2013"
      }
   ],
   "ProductInfo":{  
      "Price":null,
      "Features":null,
      "Name":null,
      "ImgURL":null,
      "ProductID":"1566954096"
   }
}
```



### Sample - 1

```
{  
   "Reviews":[  
      {  
         "Title":"Better than the park map, but do not trust it for kayaking",
         "Author":"Nicole M",
         "ReviewID":"R2JCHODG7WMHTA",
         "Overall":"3.0",
         "Content":"It is better than the NPS park map but when it comes to navigating a place like Nine Mile Pond, trust your GPS and carry a compass. It has the numbers totally inaccurate and that could cause some serious problems. They claim it is up to date, but it is not! The thing that annoys me about the folks who publish this map and field guides is they get it vetted by some knuckle head who poses as an authority but is clueless. It never occurs to them to try to use their product or watch their target market use them. So, the product is mediocre and they could care less. At one time National Geographic stood for some level of integrity but that seems to be a thing of the past.",
         "Date":"December 29, 2013"
      }
   ],
   "ProductInfo":{  
      "Price":null,
      "Features":null,
      "Name":null,
      "ImgURL":null,
      "ProductID":"1566954096"
   }
}
```

### Sample - 2

```
ormatted JSON Data
{  
   "Reviews":[  
      {  
         "Title":"Terrible product and even worse customer service",
         "Author":"Allen Gray",
         "ReviewID":"R16QAIVL4RS8EC",
         "Overall":"1.0",
         "Content":"We have purchased four Toshiba products and have eventually had problems with EVERY ONE OF THEM.  To make matters worse, their customer service is outright horrible.  They don't want to resolve issues, they just want to get you off the phone.  If you like turning blue in the face and have your blood pressure spike to near stroke levels then buy this product!",
         "Date":"February 12, 2014"
      },
      {  
         "Title":"Problem Loading MS Office",
         "Author":"Jeff Whittaker",
         "ReviewID":"R1QXLX6P9HPFIY",
         "Overall":"1.0",
         "Content":"Open office would not install.  MS Office would not install.  Not good.  I suspect MS has designed the operating system to repel Open office.",
         "Date": "August 30, 2013"
      }
   ],
   "ProductInfo":{  
      "Price":"$565.28",
      "Features":"AMD A6-4400M Accelerated Processor 2.7GHz with AMD Radeon\u2122 HD 7520G Graphics, Windows 8\n6GB DDR3 1600MHz memory, Memory Card Reader\n640GB HDD (5400rpm, Serial ATA), DVD-SuperMulti drive (+/-R double layer)\n15.6\" widescreen HD TruBrite\u00ae LED Backlit display, 1366x768 (HD), 16:9 aspect ratio, Supports 720p content\nHD Webcam and Microphone, HDMI, RGB port, Wi-Fi\u00ae Wireless networking (802.11b/g/n), Up to 4.0 hours battery life",
      "Name":"Toshiba Satellite C855D-S5135NR 16-Inch Laptop AMD A6-4400M Processor, 6GB Ram, 640GB Hard Drive, Windows 8 (Fusion Finish in Mercury Silver)",
      "ImgURL":"http://ecx.images-amazon.com/images/I/41OOpJOz4cL._SY300_.jpg",
      "ProductID":"B00B4GGZQ2"
   }
}
```

## UCSD-DataSet

## MetaData

Metadata includes 

- descriptions
- price
- sales-rank
- brand info
- co-purchasing links:

No `Rating Info`, Could be used later in the result part.

where

- `asin` - ID of the product, e.g. [0000031852](http://www.amazon.com/dp/0000031852)
- `title` - name of the product
- `price` - price in US dollars (at time of crawl)
- `imUrl` - url of the product image
- `related` - related products (also bought, also viewed, bought together, buy after viewing)
- `salesRank` - sales rank information
- `brand` - brand name
- `categories` - list of categories the product belongs to

```
{
  "asin": "0000031852",
  "title": "Girls Ballet Tutu Zebra Hot Pink",
  "price": 3.17,
  "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
  "related":
  {
    "also_bought": ["B00JHONN1S", "B002BZX8Z6", "B00D2K1M3O", "0000031909", "B00613WDTQ", "B00D0WDS9A", "B00D0GCI8S", "0000031895", "B003AVKOP2", "B003AVEU6G", "B003IEDM9Q", "B002R0FA24", "B00D23MC6W", "B00D2K0PA0", "B00538F5OK", "B00CEV86I6", "B002R0FABA", "B00D10CLVW", "B003AVNY6I", "B002GZGI4E", "B001T9NUFS", "B002R0F7FE", "B00E1YRI4C", "B008UBQZKU", "B00D103F8U", "B007R2RM8W"],
    "also_viewed": ["B002BZX8Z6", "B00JHONN1S", "B008F0SU0Y", "B00D23MC6W", "B00AFDOPDA", "B00E1YRI4C", "B002GZGI4E", "B003AVKOP2", "B00D9C1WBM", "B00CEV8366", "B00CEUX0D8", "B0079ME3KU", "B00CEUWY8K", "B004FOEEHC", "0000031895", "B00BC4GY9Y", "B003XRKA7A", "B00K18LKX2", "B00EM7KAG6", "B00AMQ17JA", "B00D9C32NI", "B002C3Y6WG", "B00JLL4L5Y", "B003AVNY6I", "B008UBQZKU", "B00D0WDS9A", "B00613WDTQ", "B00538F5OK", "B005C4Y4F6", "B004LHZ1NY", "B00CPHX76U", "B00CEUWUZC", "B00IJVASUE", "B00GOR07RE", "B00J2GTM0W", "B00JHNSNSM", "B003IEDM9Q", "B00CYBU84G", "B008VV8NSQ", "B00CYBULSO", "B00I2UHSZA", "B005F50FXC", "B007LCQI3S", "B00DP68AVW", "B009RXWNSI", "B003AVEU6G", "B00HSOJB9M", "B00EHAGZNA", "B0046W9T8C", "B00E79VW6Q", "B00D10CLVW", "B00B0AVO54", "B00E95LC8Q", "B00GOR92SO", "B007ZN5Y56", "B00AL2569W", "B00B608000", "B008F0SMUC", "B00BFXLZ8M"],
    "bought_together": ["B002BZX8Z6"]
  },
  "salesRank": {"Toys & Games": 211836},
  "brand": "Coxlures",
  "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
}
```

### 5-core

**K-cores** (i.e., dense subsets): These data have been reduced to extract the [k-core](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)), such that each of the remaining users and items have k reviews each.

where

- `reviewerID` - ID of the reviewer, e.g. [A2SUAM1J3GNN3B](http://www.amazon.com/gp/cdp/member-reviews/A2SUAM1J3GNN3B)
- `asin` - ID of the product, e.g. [0000013714](http://www.amazon.com/dp/0000013714)
- `reviewerName` - name of the reviewer
- `helpful` - helpfulness rating of the review, e.g. 2/3
- `reviewText` - text of the review
- `overall` - rating of the product
- `summary` - summary of the review
- `unixReviewTime` - time of the review (unix time)
- `reviewTime` - time of the review (raw)

```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```
### Ratings Only

**Ratings only:** These datasets include no metadata or reviews, but only (user,item,rating,timestamp) tuples. Thus they are suitable for use with [mymedialite](http://www.mymedialite.net/) (or similar) packages.

- User
- Item
- Rating
- TimeStamp

```
AKM1MP6P0OYPR,0132793040,5.0,1365811200
A2CX7LUOHB2NDG,0321732944,5.0,1341100800
A2NWSAGRHCP8N5,0439886341,1.0,1367193600
A2WNBOD3WNDNKT,0439886341,3.0,1374451200
A1GI0U4ZRJA8WN,0439886341,1.0,1334707200
A1QGNMC6O1VW39,0511189877,5.0,1397433600
A3J3BRHTDRFJ2G,0511189877,2.0,1397433600
A2TY0BTJOTENPG,0511189877,5.0,1395878400
A34ATBPOK6HCHY,0511189877,5.0,1395532800
A89DO69P0XZ27,0511189877,5.0,1395446400
```

