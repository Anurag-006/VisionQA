package com.anurag.visionqa.ui

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.anurag.visionqa.ai.MoondreamVLM
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.UUID

// --- DATA MODELS ---
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isUser: Boolean,
    val image: Bitmap? = null,
    val isLoading: Boolean = false
)

// --- HELPER FUNCTIONS ---
fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    return when (image.format) {
        ImageFormat.JPEG -> {
            val buffer: ByteBuffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }
        ImageFormat.YUV_420_888 -> {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            val nv21 = ByteArray(yBuffer.remaining() + uBuffer.remaining() + vBuffer.remaining())
            yBuffer.get(nv21, 0, yBuffer.remaining())
            vBuffer.get(nv21, yBuffer.remaining(), vBuffer.remaining())
            uBuffer.get(nv21, yBuffer.remaining() + vBuffer.remaining(), uBuffer.remaining())
            val yuvImage = android.graphics.YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
            val imageBytes = out.toByteArray()
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }
        else -> throw IllegalArgumentException("Unsupported format: ${image.format}")
    }
}

// --- UI COMPONENTS ---
@Composable
fun ShimmerBubble() {
    val shimmerColors = listOf(
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f),
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
    )
    val transition = rememberInfiniteTransition(label = "shimmer")
    val translateAnim by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1000f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Restart
        ), label = "shimmer_anim"
    )
    val brush = Brush.linearGradient(
        colors = shimmerColors,
        start = Offset(10f, 10f),
        end = Offset(translateAnim, translateAnim)
    )

    Box(
        modifier = Modifier
            .fillMaxWidth(0.6f)
            .height(50.dp)
            .background(brush, RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp))
    )
}

@Composable
fun ChatBubble(
    message: ChatMessage,
    loadingPhase: String,
    onSpeakClick: (String) -> Unit
) {
    val alignment = if (message.isUser) Alignment.End else Alignment.Start
    val bubbleColor = if (message.isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (message.isUser) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant
    val shape = if (message.isUser) {
        RoundedCornerShape(16.dp, 16.dp, 4.dp, 16.dp)
    } else {
        RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp)
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp, horizontal = 8.dp),
        horizontalAlignment = alignment
    ) {
        if (message.isLoading) {
            ShimmerBubble()
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "🧠 $loadingPhase",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
            )
        } else {
            Surface(
                color = bubbleColor,
                shape = shape,
                shadowElevation = 1.dp
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    message.image?.let { bmp ->
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "User Image",
                            modifier = Modifier
                                .height(150.dp)
                                .clip(RoundedCornerShape(8.dp)),
                            contentScale = ContentScale.Crop
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                    }

                    if (message.text.isNotEmpty()) {
                        Text(
                            text = message.text,
                            color = textColor,
                            style = MaterialTheme.typography.bodyLarge
                        )

                        // Speak Button for AI messages
                        if (!message.isUser) {
                            Spacer(modifier = Modifier.height(4.dp))
                            IconButton(
                                onClick = { onSpeakClick(message.text) },
                                modifier = Modifier.size(24.dp).align(Alignment.End)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.VolumeUp,
                                    contentDescription = "Read Aloud",
                                    tint = textColor.copy(alpha = 0.7f)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun VisionQAScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val vlm = remember { MoondreamVLM(context) }
    val speechManager = remember { SpeechManager(context) }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var questionText by remember { mutableStateOf("") }

    var messages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var isProcessing by remember { mutableStateOf(false) }
    var isInitializing by remember { mutableStateOf(true) }
    var isListening by remember { mutableStateOf(false) }

    var downloadProgress by remember { mutableStateOf("") }
    var inferenceJob by remember { mutableStateOf<Job?>(null) }
    var loadingPhase by remember { mutableStateOf("Initializing...") }

    val listState = rememberLazyListState()

    // Permission Launcher for Microphone
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            isListening = true
            scope.launch {
                speechManager.startListening().collect { text ->
                    questionText = text
                }
                isListening = false // Reset when done
            }
        }
    }

    LaunchedEffect(messages.size, messages.lastOrNull()?.text?.length) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    if (isProcessing) {
        LaunchedEffect(Unit) {
            val phases = listOf("Extracting visual features...", "Loading semantic memory...", "Analyzing prompt...", "Formulating response...")
            var i = 0
            while (true) {
                loadingPhase = phases[i % phases.size]
                i++
                delay(4000)
            }
        }
    }

    LaunchedEffect(Unit) {
        messages = listOf(ChatMessage(text = "🚀 Initializing Moondream2...\nChecking models (~1.6GB total)...", isUser = false))
        scope.launch {
            try {
                val success = vlm.initialize { msg, prog -> downloadProgress = "$msg ($prog%)" }
                messages = if (success) {
                    listOf(ChatMessage(text = "✅ Moondream is ready!\n📸 Capture an image to begin.", isUser = false))
                } else {
                    listOf(ChatMessage(text = "❌ Failed to initialize Moondream2.", isUser = false))
                }
                isInitializing = false
                downloadProgress = ""
            } catch (e: Exception) {
                messages = listOf(ChatMessage(text = "❌ Error: ${e.message}", isUser = false))
                isInitializing = false
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Box(modifier = Modifier.weight(0.35f).fillMaxWidth()) {
            if (capturedBitmap != null) {
                Image(
                    bitmap = capturedBitmap!!.asImageBitmap(),
                    contentDescription = "Captured Frame",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.3f)))
            } else {
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    onImageCaptureReady = { imageCapture = it }
                )
            }
        }

        if (downloadProgress.isNotEmpty()) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp))
            Text(
                text = downloadProgress,
                modifier = Modifier.padding(16.dp).align(Alignment.CenterHorizontally),
                style = MaterialTheme.typography.labelMedium
            )
        }

        LazyColumn(
            state = listState,
            modifier = Modifier.weight(0.65f).fillMaxWidth().padding(horizontal = 8.dp),
            contentPadding = PaddingValues(vertical = 16.dp)
        ) {
            items(messages) { msg ->
                ChatBubble(
                    message = msg,
                    loadingPhase = loadingPhase,
                    onSpeakClick = { text -> speechManager.speak(text) }
                )
            }
        }

        Surface(color = MaterialTheme.colorScheme.surface, tonalElevation = 4.dp, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = questionText,
                        onValueChange = { questionText = it },
                        label = { Text("Ask about the image...") },
                        modifier = Modifier.weight(1f),
                        enabled = !isProcessing && !isInitializing,
                        maxLines = 3,
                        shape = RoundedCornerShape(16.dp),
                        trailingIcon = {
                            // MICROPHONE BUTTON
                            IconButton(
                                onClick = {
                                    if (isListening) {
                                        isListening = false // Stop visually
                                        // The flow will complete naturally
                                    } else {
                                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                    }
                                }
                            ) {
                                Icon(
                                    imageVector = if (isListening) Icons.Default.Stop else Icons.Default.Mic,
                                    contentDescription = "Voice Input",
                                    tint = if (isListening) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                if (isProcessing) {
                    Button(
                        onClick = {
                            inferenceJob?.cancel()
                            isProcessing = false
                            messages = messages.map {
                                if (it.isLoading) it.copy(isLoading = false, text = "⚠️ Generation stopped by user.") else it
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                    ) {
                        Text("Stop Generation")
                    }
                } else {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            modifier = Modifier.weight(1f),
                            enabled = !isInitializing,
                            onClick = {
                                if (capturedBitmap != null) {
                                    capturedBitmap = null
                                    vlm.resetChat()
                                    messages = emptyList()
                                } else {
                                    val capture = imageCapture ?: return@Button
                                    capture.takePicture(
                                        ContextCompat.getMainExecutor(context),
                                        object : ImageCapture.OnImageCapturedCallback() {
                                            override fun onCaptureSuccess(image: ImageProxy) {
                                                try {
                                                    capturedBitmap = imageProxyToBitmap(image)
                                                } finally {
                                                    image.close()
                                                }
                                            }
                                            override fun onError(exc: ImageCaptureException) {
                                                messages = messages + ChatMessage(text = "❌ Camera error: ${exc.message}", isUser = false)
                                            }
                                        }
                                    )
                                }
                            }
                        ) {
                            Text(if (capturedBitmap != null) "Retake Image" else "Capture")
                        }

                        Button(
                            modifier = Modifier.weight(1f),
                            enabled = capturedBitmap != null && questionText.isNotBlank(),
                            onClick = {
                                val bitmap = capturedBitmap!!
                                val question = questionText.trim()

                                questionText = ""
                                isProcessing = true

                                val isFirstQuestion = messages.none { it.isUser }
                                messages = messages + ChatMessage(text = question, isUser = true, image = if (isFirstQuestion) bitmap else null)

                                val aiMsgId = UUID.randomUUID().toString()
                                messages = messages + ChatMessage(id = aiMsgId, text = "", isUser = false, isLoading = true)

                                inferenceJob = scope.launch {
                                    try {
                                        val answer = vlm.chat(bitmap, question, null, emptyList()) { token ->
                                            messages = messages.map {
                                                if (it.id == aiMsgId) it.copy(isLoading = false, text = it.text + token) else it
                                            }
                                        }
                                        messages = messages.map {
                                            if (it.id == aiMsgId && it.text.isEmpty()) it.copy(isLoading = false, text = answer) else it
                                        }
                                    } catch (e: CancellationException) {
                                    } catch (e: Exception) {
                                        messages = messages.map {
                                            if (it.id == aiMsgId) it.copy(isLoading = false, text = "❌ Error: ${e.message}") else it
                                        }
                                    } finally {
                                        isProcessing = false
                                    }
                                }
                            }
                        ) {
                            Text("Send")
                        }
                    }
                }
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            vlm.cleanup()
            speechManager.cleanup() // Cleanup speech resources
        }
    }
}