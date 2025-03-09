package com.example.combine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.content.FileProvider
import com.example.combine.ui.theme.CombineTheme
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.text.method.ScrollingMovementMethod // Add this line
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : ComponentActivity() {


    private lateinit var cameraImage: ImageView
    private lateinit var captureImgBtn: Button
    private lateinit var resultText: TextView
    private lateinit var processBtn: Button
    private lateinit var classifiedResult: TextView

    private var currentPhotoPath: String? = null
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>

    private lateinit var interpreter: Interpreter
    private lateinit var vocab: Map<String, Int>
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraImage = findViewById(R.id.cameraImage)
        captureImgBtn = findViewById(R.id.captureImgBtn)
        resultText = findViewById(R.id.resultText)
        processBtn = findViewById(R.id.processBtn)
        classifiedResult = findViewById(R.id.classifiedResult)

        resultText.movementMethod = ScrollingMovementMethod()

        // Initialize model and vocab

        val modelPath= "model_main.tflite"
        interpreter = Interpreter(loadModel(this, modelPath))
        loadVocab()
        loadLabels()

        // Permissions and image capture setup
        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
                isGranted ->
            if (isGranted) {
                captureImage()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

        takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                currentPhotoPath?.let { path ->
                    val bitmap = BitmapFactory.decodeFile(path)
                    cameraImage.setImageBitmap(bitmap)
                    recognizeText(bitmap)
                }
            }
        }

        captureImgBtn.setOnClickListener {
            requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }

        processBtn.setOnClickListener {
            val extractedText = resultText.text.toString()
            if (extractedText.isNotBlank()) {
                classifyText(extractedText)
            } else {
                Toast.makeText(this, "No text available for classification", Toast.LENGTH_SHORT).show()
            }
        }

    }

    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun captureImage() {
        val photoFile: File? = try {
            createImageFile()
        } catch (ex: IOException) {
            Toast.makeText(this, "Error occurred while creating the file", Toast.LENGTH_SHORT).show()
            null
        }
        photoFile?.also {
            val photoUri: Uri = FileProvider.getUriForFile(this, "${applicationContext.packageName}.provider", it)
            takePictureLauncher.launch(photoUri)
        }
    }

    private fun recognizeText(bitmap: Bitmap) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        recognizer.process(image).addOnSuccessListener { ocrText ->
            resultText.text = ocrText.text
            Toast.makeText(this, "Text recognized successfully", Toast.LENGTH_SHORT).show()
        }.addOnFailureListener { e ->
            Toast.makeText(this, "Failed to recognize text: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun classifyText(text: String) {
        val inputVector = preprocessText(text)
        val inputBuffer = ByteBuffer.allocateDirect(inputVector.size * 4).order(ByteOrder.nativeOrder())
        inputVector.forEach { inputBuffer.putFloat(it) }

        val outputBuffer = ByteBuffer.allocateDirect(labels.size * 4).order(ByteOrder.nativeOrder())
        interpreter.run(inputBuffer, outputBuffer)

        val outputArray = FloatArray(labels.size)
        outputBuffer.rewind()
        for (i in outputArray.indices) {
            outputArray[i] = outputBuffer.float
        }

        val maxIndex = outputArray.indices.maxByOrNull { outputArray[it] } ?: -1
        val predictedLabel = if (maxIndex >= 0) labels[maxIndex] else "Unknown"
        classifiedResult.text = "Classification Result: \n$predictedLabel"
    }

    private fun preprocessText(text: String): FloatArray {
        val tokens = text.lowercase().split(Regex("\\s+")).map { it.replace(Regex("[^a-z0-9]"), "") }
        val featureVector = FloatArray(1000) { 0f }

        for (token in tokens) {
            val index = vocab[token]
            if (index != null && index < 1000) {
                featureVector[index] += 1f
            }
        }
        return featureVector
    }


    private fun loadModel(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
        }
    }


    private fun loadVocab() {
        val vocabJson = assets.open("tfidf_vocab.json").bufferedReader().use { it.readText() }

        // Parse the entire JSON object
        val fullJson: Map<String, Any> = Gson().fromJson(vocabJson, object : TypeToken<Map<String, Any>>() {}.type)

        // Extract "vocab" safely and convert to Map<String, Int>
        val extractedVocab = (fullJson["vocab"] as? Map<String, Number>) ?: emptyMap()
        vocab = extractedVocab.mapValues { it.value.toInt() }  // Ensure conversion to Int

        // Extract "idf" safely (if needed) and convert to List<Float>
        val extractedIdf = (fullJson["idf"] as? List<Number>)?.map { it.toFloat() } ?: emptyList()
    }




    private fun loadLabels() {
        val labelsJson = assets.open("labels.json").bufferedReader().use { it.readText() }
        labels = Gson().fromJson(labelsJson, object : TypeToken<List<String>>() {}.type)
    }



    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }
}

