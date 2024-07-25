# Proyecto Final de Grado
## Estudio y Mejora de Técnicas de Segmentación de Imágenes a través del Aprendizaje Auto Supervisado
**Autor:** Ismael Tse Perdomo Rodríguez
**Tutores:** Félix Fuentes y Guillermo Iglesias

### Asbtract 
Pese a los excelentes resultados obtenidos en el campo del procesamiento del lenguaje natural y la clasificación de imágenes,
el paradigma Auto Supervisado no ha tenido especial relevancia respecto a la tarea de segmentación semántica. El presente
Proyecto de Fin de Grado (PFG) se centra en el estudio y mejora de las técnicas actuales de segmentación de imágenes laparoscópicas,
utilizando el aprendizaje auto supervisado, con el objetivo de mejorar los resultados clínicos y la seguridad de los pacientes.

Para ello, se ha implementado un *pipeline* de pre-entrenamiento auto supervisado, donde se combinan métodos discriminativos
y generativos para extraer información relevante de la naturaleza de las imágenes, sin la necesidad de extraer etiquetas manualmente.
Este enfoque ha sido validado mediante experimentos exhaustivos, comparando su rendimiento con dos *baselines* supervisados,
que sirven a modo de representantes de las principales arquitecturas tradicionales.
    
Los resultados obtenidos demuestran una mejora significativa en la segmentación de imágenes laparoscópicas, con una reducción drástica
de hasta el 40% en el tiempo de procesamiento comparado con las técnicas supervisadas. Estas mejoras son especialmente notables en
escenarios clínicos complejos donde la variabilidad de las imágenes es alta y el margen de error es mínimo, alcanzando un *Dice Coefficient*
DC de 0.95. De este modo, se abre la posibilidad de aplicar esta metodología en entornos clínicos reales, para continuar avanzando en
la precisión y eficiencia de los procedimientos quirúrgicos asistidos por imagen.

### Citations
```bibtex
@mastersthesis{citekey,
  title = {Estudio y Mejora de Técnicas de Segmentación en Imágenes Laparoscópicas
  a través del Aprendizaje Auto Supervisado},
  author = {Ismael Tse Perdomo y Felix Fuentes y Guillermo Iglesias},
  school = {E.T.S. de Ingeniería de Sistemas Informáticos},
  year = {2024},
  month = {7},
  type = {Proyecto Fin de Grado}
}
```
