# Documentazione Tecnica Python

Questa sezione raccoglie esempi e spiegazioni più **tecniche** per l'uso di Python.

---

## Struttura di un modulo Python
Un modulo è un file `.py` che contiene funzioni, classi e variabili.

```python
# math_utils.py
def somma(a, b):
    # Restituisce la somma di due numeri
    return a + b

def moltiplica(a, b):
    # Restituisce il prodotto di due numeri
    return a * b
```

Uso del modulo:
```python
import math_utils

print(math_utils.somma(2, 3))      # 5
print(math_utils.moltiplica(2, 3)) # 6
```

---

## Tipizzazione statica con `typing`
Python supporta **hinting** per una maggiore leggibilità e supporto da parte degli IDE.

```python
from typing import List

def media(valori: List[float]) -> float:
    return sum(valori) / len(valori)

print(media([2.5, 3.5, 4.0]))
```

---

## Documentazione con docstring
Le docstring sono stringhe di testo che spiegano funzioni, classi e moduli.

```python
def divisione(a: float, b: float) -> float:
    '''
    Esegue la divisione di due numeri.

    Args:
        a (float): Dividendo.
        b (float): Divisore.

    Returns:
        float: Risultato della divisione.

    Raises:
        ZeroDivisionError: Se `b` è uguale a zero.
    '''
    return a / b
```

---

## Testing con `unittest`
```python
import unittest
from math_utils import somma

class TestMathUtils(unittest.TestCase):
    def test_somma(self):
        self.assertEqual(somma(2, 3), 5)

if __name__ == "__main__":
    unittest.main()
```
