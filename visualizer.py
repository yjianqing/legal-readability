import numpy as np
from xml.etree import ElementTree as ET

def print_example(inputs, scores, prediction, label):
    print()
    for n, token in enumerate(str(inputs).split()):
        print(scores[n], token)
    print(prediction, label)

def create_viz_doc():
    html = ET.Element('html')
    head = ET.SubElement(html, 'head')
    css = ET.SubElement(head, 'link')
    css.set('rel', 'stylesheet')
    css.set('type', 'text/css')
    css.set('href', 'visualizer.css')
    body = ET.SubElement(html, 'body')
    doc = ET.ElementTree(element=html)
    return doc

def activation(score):
    return 'a' + str(int(np.round(score * 10)) + 10)

def visualize_example(doc, inputs, scores, states, prediction, label):
    body = doc.find('body')
    example = ET.SubElement(body, 'div')
    example.set('class', 'example')
    #Overall prediction and label
    overall = ET.SubElement(example, 'div')
    span = ET.SubElement(overall, 'span')
    span.text = 'Prediction:'
    span = ET.SubElement(overall, 'span')
    span.text = str(np.around((prediction + 1) / 2, decimals=3))
    span.set('class', activation(prediction))
    span = ET.SubElement(overall, 'span')
    span.text = 'Label:'
    span = ET.SubElement(overall, 'span')
    span.text = str(label)
    span.set('class', activation(label - pow(0, label)))
    #Token prediction
    div = ET.SubElement(example, 'div')
    for n, token in enumerate(str(inputs).split()):
        span = ET.SubElement(div, 'span')
        span.text = token
        span.set('class', activation(scores[n]))
    #Token prediction by hidden units
    div = ET.SubElement(example, 'div')
    div.text = 'Predictions by hidden units:'
    for layer in range(len(states[0])):
        div = ET.SubElement(example, 'div')
        for n, token in enumerate(str(inputs).split()):
            span = ET.SubElement(div, 'span')
            span.text = token
            span.set('class', activation(states[n][layer]))

def write_viz_doc(doc):
    doc.write('activation.html')