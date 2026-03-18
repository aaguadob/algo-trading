# Arquitectura


```
                +------------------+
                |  Configuration   |
                +------------------+

                        ↓

+---------+     +------------+     +------------+
|  Data   | →   |  Features  | →   |  Strategy  |
+---------+     +------------+     +------------+
                                       ↓
                                  +-----------+
                                  | Portfolio |
                                  +-----------+
                                       ↓
                                  +-----------+
                                  |   Risk    |
                                  +-----------+
                                       ↓
                                  +-----------+
                                  | Execution |
                                  +-----------+
```


```
MarketEvent
   ↓
Strategy
   ↓
SignalEvent
   ↓
Portfolio
   ↓
OrderEvent
   ↓
Execution
   ↓
FillEvent
   ↓
Portfolio
```

