# Aspetti Avanzati di Python

Dopo le basi, Python offre molte funzionalità avanzate che permettono di scrivere codice più **flessibile ed efficiente**.

## Programmazione Orientata agli Oggetti (OOP)
Esempio di classe:
```python
class Animale:
    def __init__(self, nome):
        self.nome = nome
    
    def parla(self):
        raise NotImplementedError("Questo metodo deve essere implementato dalle sottoclassi.")

class Cane(Animale):
    def parla(self):
        return "Bau!"

rex = Cane("Rex")
print(rex.parla())  # Output: Bau!
```

## Funzioni di ordine superiore
```python
def saluta(nome):
    return f"Ciao, {nome}!"

def esegui(funzione, argomento):
    return funzione(argomento)

print(esegui(saluta, "Alice"))
```

## Decoratori
I decoratori permettono di modificare il comportamento delle funzioni.
```python
def log(funzione):
    def wrapper(*args, **kwargs):
        print(f"Esecuzione di {funzione.__name__}")
        return funzione(*args, **kwargs)
    return wrapper

@log
def somma(a, b):
    return a + b

print(somma(3, 4))
```

## Gestione delle eccezioni
```python
try:
    x = int("abc")
except ValueError as e:
    print(f"Errore: {e}")
```