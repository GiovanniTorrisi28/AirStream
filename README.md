# AirStream 

## Descrizione
Questo progetto è un software per l'analisi della qualità dell'aria con riferimento alla città di Catania (Sicilia, Italia).

I dati dell'inquinamento vengono acquisiti in streaming da una sorgente esterna vengono elaborati in tempo reale con lo scopo di: 
- Monitorare l'andamento della qualità dell'aria e le principali condizioni atmosferiche
- Predire la qualità dell'aria per l'ora successiva

La qualità dell'aria è rappresentata da un indice numerico intero (aqi) che varia da 1 a 5:
- 1 → Altà qualità
- 5 → Elevato grado di inquinamento

## Setup
Per eseguire il progetto in locale è necessario avere installato **Docker** sul proprio host. (https://www.docker.com/get-started)

Dopo aver installato Docker:

```bash
# Clonare il repository
git clone https://github.com/GiovanniTorrisi28/AirStream

# Spostarsi nella directory del progetto
cd AirStream

# Avviare i container
docker-compose up -d
```

Una volta avviato l'ambiente, è possibile accedere a Kibana per la visualizzazione dei dati all'indirizzo http://localhost:5601


## Tecnologie e Infrastruttura

La sorgente dei dati sull'inquinamento atmosferico è Open Weather Map. I dati sono accessibili tramite REST API all'indirizzo: https://openweathermap.org 

Il progetto è realizzato tramite un'architettura a micro-servizi basata su Docker per agevolare la portabilità dell'applicazione.Sono state quindi utilizzate le immagini dei seguenti concomponenti: 
- Logstash: si occupa di fare **ingestion** dei dati provenienti dalla sorgente, effettuando polling a intervalli regolari di un minuto e inviando i dati a kafka, il sistema di messaggistica adottato.
- Kafka: gestisce i dati in streaming tramite un topic dedicato. Kafka garantisce alta affidabilità e permette di disaccoppiare il momento in cui i dati vengono acquisiti da quello in cui vengono elaborati, garantendo scalabilità e servizio di storage temporaneo.
- Spark: Effettua l'elaborazione dei dati e applica un modello di machine learning per stimare la qualità dell'aria per l'ora successiva. Il task individuato è quello della classificazione multiclasse nei {1,2,3,4,5}. Il modello scelto è Random Forest Classifier.

- Elasticsearch: indicizza e memorizza i dati elaborari, consentendo query efficienti e fungendo da storga finale.

- Kibana: Fornisce strumenti di visualizzazione interattiva dei dati indicizzati su ElasticSearch, permettendo la creazione di dashboard dinamiche che mostrano grafici temporali, heatmap, mappe, indicatori ecc..

## Struttura del Repository

```plaintext
AirStream/
│── docker-compose.yml        # Avvio dei container
│
├── logstash/
│   └── pipeline/
│       └── logstash.conf     # Configurazione per l’ingestion dei dati
│
├── spark/
│   ├── Dockerfile            # File per la costruzione dell' immagine Spark con le librerie necessarie
│   ├── models/               # Modello Random Forest addestrato
│   ├── dataset/              # File CSV per training e validation
│   └── src/                  # Codici Python (training, validazione, script principale)
│
├── elasticsearch/
│   ├── esdata/               # Dati persistenti salvati su Elasticsearch
│   
└── README.md                 # Documentazione del progetto
