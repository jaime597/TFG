# TFG

Dentro de la carpeta code, se hallan los siguientes ficheros:

- rmskmeans.py, archivo encargado de aplicar el algoritmo random min size kmeans basado en el min size kmeans. Trabaja con el csv dataset_final.csv
- prmskmeans.py está la versión paralelizada usando Pandarallel. Esta versión no es muy estable, y falla
  en ciertos sistemas.

- mskmeansrn.ipynb notebook para ejecutar rmskmeans.
- generate_superclusters.ipynb, algoritmo encargado de generar 2 nuevas columnas clúster con la mitad de grupos dada la original. Es decir, 100 --> 50 --> 25
- dataset2.ipynb se encuentra el código principal, con la GUI y el algoritmo. 

Se puede ejecutar dataset2 directamente ya que se han subido los archivos necesarios para ello. No obstante, si se prefiere se puede ejecutar mskmeansrn para crear la primera clusterización, después, generate_superclusters para los dos siguientes niveles y, finalmente, dataset2. 

En cuanto a la GUI se recomienda lo siguiente:
  Se debe seleccionar csv/definitivo.csv como el dataset deseado.
  A continuación, se muestra el csv y una vez se ha visualizado el tiempo deseado se hace click en la 'X' de la ventana.
  Ahora se deben seleccionar los pseudoidentificadores del dataset (sexo, fecha_nac, cias_cd, ine11_cd, codpos, cluster y neighbour_cd).
  Por último, seleccionar jerarquías y niveles:
  - sexo: Género, Nivel 0
  - fecha_nac: Fechas, Nivel 0
  - ine11_cd: ID municipio, Nivel 0
  - codpos: Zona, Nivel 0
  - cluster: Ubicación geográfica, Nivel 0
  - neighbour_cd: ID municipio, Nivel 1

Se recomienda seleccionar lo indicado, y si no al menos todos los atributos correspondientes al Nivel 0. 
Todo aquello que no se seleccione no se podrá generar (niveles más globales). Por ende, los resultados que salgan serán más pobres o incluso puede haber errores de falta de jerarquías.

Adicionalmente, se ha implementado para si se tiene algún problema o no se quiere rellenar todo esto, que si se presiona el botón Cancel
de la primera ventana, el algoritmo funciona de manera automática (se le pasan manualmente las jerarquías y niveles).

En las carpetas se halla lo siguiente:
  - csv: almacena los archivos necesarios para que funcione el programa principal (dataset principal y datasets a los que hacer join). Estos se generan también al    ejecutar generate_superclusters, a excepción de cias_zona_sector.csv (almacena la información de niveles de salud de Aragón) y dataset_final.csv que es el dataset completo de trabajo.
  - def: en esta carpeta se generan los archivos de mskmeansrn.ipynb. La carpeta es necesario que exista, si no no podrá guardarlo y generate_superclusters no podrá leer los archivos.
  
Tener en cuenta que los nombres y archivos pueden ser cambiados por otros, pero tienen que cumplir la estructura establecida.

dataset_final.csv es distinto al que se empleó para hacer las pruebas finales que aparecen en la memoria, ambos dos son similares pero no iguales, de ahí que los resultados varíen. A este se le han modificado datos para que no sea posible identificar a ningún sujeto.
