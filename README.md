# Age Group Multi-Class Classification using Neural Network

Tento projekt zahŕňa vybudovanie a tréning neurónovej siete na klasifikáciu vekových skupín na základe údajov o zákazníkoch. Projekt využíva PyTorch na vytváranie modelov a školenia a zahŕňa funkcie predspracovania údajov, vyhodnocovania modelov a predikcie.

## Obsah
- [Prehľad projektu](#prehľad-projektu)
- [Požiadavky](#požiadavky)
- [Údaje](#údaje)
- [Model](#model)
- [Tréning](#tréning)
- [Hodnotenie](#hodnotenie)
- [Predikcie](#predikcie)
- [Použitie](#použitie)
- [Licencia](#licencia)

## Prehľad projektu
Cieľom tohto projektu je klasifikovať vekové skupiny zákazníkov do vopred definovaných kategórií:
- Youth (<25)
- Young Adults (25-34)
- Adults (35-64)
- Seniors (64+)

Zahŕňa:
- Predspracovanie a kódovanie údajov
- Definícia architektúry neurónovej siete
- Trenovani a hodnotenie modelu
- Predpovedanie nových údajov

## Požiadavky
- Python 3.x
- PyTorch
- Pandy
- NumPy
- Matplotlib
- Scikit-učte sa

Požadované balíčky môžete nainštalovať pomocou pip:

``` bash
pip install pochodeň pandy numpy matplotlib scikit-learn
```
## Údaje
Súbor údajov použitý v tomto projekte obsahuje nasledujúce stĺpce:

Customer_Age,
Customer_Gender,
Country,
State,
Product_Category,
Order_Quantity,
Profit,
Revenue.

## Model
### Architektúra neurónovej siete
Model neurónovej siete AgeGroupNN je definovaný s nasledujúcimi vrstvami:

Vstupná vrstva,
Dve skryté vrstvy s normalizáciou a dropout,
Výstupná vrstva so 4 neuronmi zodpovedajúcimi vekovým skupinám.
``` bash
class AgeGroupNN(nn.Module):
    def __init__(self, in_features=8, hl1=80, hl2=60, hl3=20, out_features=4):
        super(AgeGroupNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hl1)
        self.fc3 = nn.Linear(hl1, hl2)
        self.bn3 = nn.BatchNorm1d(hl2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(hl2, hl3)
        self.bn4 = nn.BatchNorm1d(hl3)
        self.dropout4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(hl3, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)  # (zakomentované) aplikácia dropout
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x
```
## Tréning
### Tréningový proces
Tréningový proces zahŕňa:

Tréning modelu pre určený počet epoch,
Implementácia early stoping, aby sa zabránilo nadmernému overfitting,
Sledovanie strát a presností treningu a hodnotenia.
``` bash
def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    criterion, 
    optimizer, 
    num_epochs = 100,
    patience: int = 10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Training code here
```
## Hodnotenie
### Metriky hodnotenia
Výkonnosť modelu sa hodnotí pomocou metrík straty a presnosti pre tréningové aj validacne súbory údajov. Výsledky sú vizualizované pomocou matplotlib.
``` bash
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    # Plotting code here
```
## Predpoveď
### Funkcia predpovedania
Funkcia predict_age_group vám umožňuje predpovedať nové údaje.
``` bash
def predict_age_group(model, input_data, scaler, ordinal_encoder):
    # Prediction code here
```
### Predikcie
Môžete vygenerovať náhodné vzorové údaje na testovanie funkcie predikcie.
``` bash
def generate_input_data():
    # Data generation code here
```
## Použitie
1. Generovanie vstupných údajov:
``` bash
input_data = generate_input_data()
```
2. Predikcie:
``` bash
predictions = predict_age_group(model, input_data, scaler, ordinal_encoder)
```
3. Vizualizujte výsledky:
``` bash
plot_results(train_losses, train_accuracies, val_losses, val_accuracies)
```
## Licencia
Tento projekt je licencovaný pod licenciou MIT – podrobnosti nájdete v súbore LICENCIA.
