# Machinelles Lernen auf verteilten System zur Bildklassifizierung mit TensorFlow

## Inhalte
Dieses Repository beinhalted jeglichen benutzen Quellcode der in der Bachelorarbeit "Machinelles Lernen auf verteilten System zur Bildklassifizierung mit TensorFlow" verwendet wurde.
Dazu zählen die in AWS verwendeten Notebooks in Skripte als auch mehrere beispiele für ein Setup außerhalb einer cloud mittels mehreren Computern und die erlangten Testergebnisse.

### Beispiel
In diesem Verzeichnis befindet sich ein Beispiel wie man TensorFlow verteilt konfigurieren kann.
Dabei wird sowohl die synchrone Multi Worker Mirrored Strategy als auch die asynchrone Paramester Server Strategy. 
Bei beiden Beispiel wird dafür der Mnist Datensatz geladen und in einem Netz mit einer dense Layer trainiert.

### AWS
Im AWS Verzeichnis befinden sich der source code mit dem das verteilte Training in AWS getestet wurde. Dafür gibt es lediglich ein Notebook welche 3 verschiedene TensorFlow Estimator für unverteiltes, synchrones und asynchrones  training enthalten.
Für die verschiedenen Tests wurden lediglich die Parameter wie Instance Count, Batch Size und Epochs verändert.

### Testergebnisse
Sämtliche Tests und ihre Ergebnisse befinden sich in folgender tabelle:
https://docs.google.com/spreadsheets/d/1krebaLqn6xySOQoYF6woLxtLSYR_gDziE8wJghgJ0No/edit#gid=0
Die angegebene test_acc, val_acc, train_acc usw. beziehen sich hierbei auf die zuletzt errechneten Werte nach der finalen Epoche.
Zusätzlich befinden sich im tensorboard_logs-Verzeichnis die erstellten Graphen. Um ein Tensorboard für dieses Verzeichnis zu hosten kann folgender Befehl verwendet werden sofarn TensorFlow installiert ist.
$ tensorboard --logdir=aws/tensorboard_logs
Die Logs sind hier mit ihren Jobnamen betitelt. Das Schema hierbei ist folgende.

MWM001-33000d-artType-2i-128b-m5-4xlarge-10e-230713-102652

MWM steht für Multi Worker Mirrored Strategy
201 2 Steht für die Testgruppe, 1 für die Testnummer innerhalb der Gruppe.
33000d ist die Größe des verwendeten Datensatzes.
artType steht für article Type als verwendete Klassifikation.
2i ist die Instanzanzahl
128b die Batch Size
m5-4xlarge ist der verwendete AWS Instanztyp
10e die Anzahl der Epochen
Die letzten Zahlen sind ein timestamp