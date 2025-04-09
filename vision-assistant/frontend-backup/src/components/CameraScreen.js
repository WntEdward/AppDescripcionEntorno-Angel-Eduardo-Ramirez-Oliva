import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as Speech from 'expo-speech';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

// TensorCamera envuelve la cámara normal para procesamiento con TFJS
const TensorCamera = cameraWithTensors(Camera);

export default function CameraScreen() {
  const [hasPermission, setHasPermission] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isTalking, setIsTalking] = useState(false);
  const cameraRef = useRef(null);
  const websocket = useRef(null);

  // Pedir permisos de cámara
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
      
      // Inicializar TensorFlow.js
      await tf.ready();
      console.log('TensorFlow está listo');
    })();
  }, []);

  // Configurar WebSocket
  useEffect(() => {
    // Reemplaza con la URL de tu backend
    const wsUrl = 'ws://tu-backend:8000/ws/detect';
    websocket.current = new WebSocket(wsUrl);

    websocket.current.onopen = () => {
      console.log('Conectado al servidor WebSocket');
    };

    websocket.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.success) {
        setDetections(data.detections);
        describeObjects(data.detections);
      }
    };

    return () => {
      if (websocket.current) {
        websocket.current.close();
      }
    };
  }, []);

  const describeObjects = (objects) => {
    if (objects.length === 0 || isTalking) return;
    
    setIsTalking(true);
    const topObjects = objects
      .filter(obj => obj.confidence > 0.6)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    if (topObjects.length > 0) {
      const description = topObjects.map(obj => 
        `${obj.class} con ${Math.round(obj.confidence * 100)}% de certeza`
      ).join('. ');

      Speech.speak(description, {
        language: 'es',
        rate: 0.9,
        onDone: () => setIsTalking(false),
        onStopped: () => setIsTalking(false),
      });
    } else {
      setIsTalking(false);
    }
  };

  const handleCameraStream = async (images) => {
    const nextImageTensor = images.next().value;
    
    if (!nextImageTensor || !websocket.current || websocket.current.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      // Convertir tensor a JPEG
      const imageTensor = nextImageTensor;
      const imageData = await tf.browser.toPixels(imageTensor);
      
      // Crear canvas temporal para convertir a JPEG
      const canvas = document.createElement('canvas');
      canvas.width = imageTensor.shape[1];
      canvas.height = imageTensor.shape[0];
      const ctx = canvas.getContext('2d');
      
      const imageDataClamped = new Uint8ClampedArray(imageData);
      const imgData = new ImageData(imageDataClamped, canvas.width, canvas.height);
      ctx.putImageData(imgData, 0, 0);
      
      // Convertir a blob y enviar
      canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onload = () => {
          if (reader.readyState === 2) {
            const buffer = new Uint8Array(reader.result);
            websocket.current.send(buffer);
          }
        };
        reader.readAsArrayBuffer(blob);
      }, 'image/jpeg', 0.7); // 70% de calidad para reducir tamaño
      
    } catch (error) {
      console.error('Error procesando frame:', error);
    } finally {
      tf.dispose(nextImageTensor);
    }
  };

  if (hasPermission === null) {
    return <View style={styles.container}><Text>Solicitando permisos...</Text></View>;
  }
  if (hasPermission === false) {
    return <View style={styles.container}><Text>Sin acceso a la cámara</Text></View>;
  }

  return (
    <View style={styles.container}>
      <TensorCamera
        ref={cameraRef}
        style={styles.camera}
        type={Camera.Constants.Type.back}
        resizeWidth={320}
        resizeHeight={240}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
      />
      
      <View style={styles.detectionsContainer}>
        {detections.map((obj, idx) => (
          <Text key={idx} style={styles.detectionText}>
            {obj.class} ({Math.round(obj.confidence * 100)}%)
          </Text>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  detectionsContainer: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 5,
  },
  detectionText: {
    color: 'white',
    fontSize: 16,
    marginVertical: 2,
  },
});