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
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.anurag.visionqa.ai.GemmaVLM
import com.anurag.visionqa.ai.OcrHelper
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.UUID

// ── Palette ──────────────────────────────────────────────────────────────────
// Clean, professional dark-leaning palette with a distinctive teal/indigo accent
private val BgDeep       = Color(0xFF0F1117)   // near-black background
private val BgCard       = Color(0xFF1A1D27)   // card surface
private val BgInput      = Color(0xFF252837)   // input field background
private val BgOverlay    = Color(0xFF1E2130)   // subtle overlay

private val AccentTeal   = Color(0xFF00C9A7)   // primary accent — vibrant teal
private val AccentIndigo = Color(0xFF7B6FF0)   // secondary accent — soft indigo
private val AccentAmber  = Color(0xFFFFB830)   // warning/status accent

private val UserBubble   = Color(0xFF1E3A5F)   // user message bg
private val AiBubble     = Color(0xFF1A1D27)   // ai message bg (same as card)

private val TextPrimary  = Color(0xFFF0F2F8)   // main text
private val TextSecond   = Color(0xFF8A8FA8)   // secondary text
private val TextCode     = Color(0xFF00C9A7)   // code text (matches accent)
private val DividerColor = Color(0xFF2A2D3E)

// ── Data ─────────────────────────────────────────────────────────────────────
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isUser: Boolean,
    val image: Bitmap? = null,
    val isLoading: Boolean = false
)

// ── Image util ───────────────────────────────────────────────────────────────
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

// ── Markdown renderer ─────────────────────────────────────────────────────────
@Composable
fun MarkdownText(
    text: String,
    modifier: Modifier = Modifier,
    baseColor: Color = TextPrimary
) {
    val lines = text.split("\n")
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        lines.forEach { line ->
            when {
                line.startsWith("# ") -> {
                    Text(
                        text = parseInline(line.removePrefix("# "), baseColor),
                        style = TextStyle(
                            fontSize = 19.sp,
                            fontWeight = FontWeight.Bold,
                            color = AccentTeal,
                            lineHeight = 27.sp
                        )
                    )
                }
                line.startsWith("## ") -> {
                    Text(
                        text = parseInline(line.removePrefix("## "), baseColor),
                        style = TextStyle(
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = AccentIndigo,
                            lineHeight = 23.sp
                        )
                    )
                }
                line.startsWith("### ") -> {
                    Text(
                        text = parseInline(line.removePrefix("### "), baseColor),
                        style = TextStyle(
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = AccentAmber,
                            lineHeight = 21.sp
                        )
                    )
                }
                line.trim() == "---" || line.trim() == "***" -> {
                    HorizontalDivider(
                        color = DividerColor,
                        thickness = 1.dp,
                        modifier = Modifier.padding(vertical = 4.dp)
                    )
                }
                line.startsWith("- ") || line.startsWith("* ") -> {
                    Row(
                        modifier = Modifier.padding(start = 8.dp),
                        verticalAlignment = Alignment.Top,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(5.dp)
                                .offset(y = 9.dp)
                                .background(AccentTeal, CircleShape)
                        )
                        Text(
                            text = parseInline(line.removePrefix("- ").removePrefix("* "), baseColor),
                            style = TextStyle(fontSize = 14.sp, lineHeight = 22.sp, color = baseColor)
                        )
                    }
                }
                line.matches(Regex("^\\d+\\.\\s.*")) -> {
                    val num = line.substringBefore(".").trim()
                    val content = line.substringAfter(". ")
                    Row(
                        modifier = Modifier.padding(start = 8.dp),
                        verticalAlignment = Alignment.Top,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "$num.",
                            style = TextStyle(
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold,
                                color = AccentTeal
                            ),
                            modifier = Modifier.width(22.dp)
                        )
                        Text(
                            text = parseInline(content, baseColor),
                            style = TextStyle(fontSize = 14.sp, lineHeight = 22.sp, color = baseColor)
                        )
                    }
                }
                line.startsWith("    ") || line.startsWith("\t") -> {
                    Surface(
                        color = Color(0xFF0D1117),
                        shape = RoundedCornerShape(6.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = line.trimStart(),
                            style = TextStyle(
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                color = TextCode,
                                lineHeight = 18.sp
                            ),
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                        )
                    }
                }
                line.isBlank() -> Spacer(modifier = Modifier.height(2.dp))
                else -> {
                    Text(
                        text = parseInline(line, baseColor),
                        style = TextStyle(fontSize = 14.sp, lineHeight = 22.sp, color = baseColor)
                    )
                }
            }
        }
    }
}

fun parseInline(text: String, baseColor: Color) = buildAnnotatedString {
    val bold   = SpanStyle(fontWeight = FontWeight.Bold, color = baseColor)
    val italic = SpanStyle(fontStyle = FontStyle.Italic, color = baseColor.copy(alpha = 0.85f))
    val code   = SpanStyle(
        fontFamily = FontFamily.Monospace,
        background = Color(0xFF0D1117),
        color = AccentTeal,
        fontSize = 12.sp
    )
    val strike = SpanStyle(textDecoration = TextDecoration.LineThrough, color = baseColor.copy(alpha = 0.5f))

    var i = 0
    while (i < text.length) {
        when {
            text.startsWith("**", i) -> {
                val end = text.indexOf("**", i + 2)
                if (end != -1) { withStyle(bold) { append(text.substring(i + 2, end)) }; i = end + 2 }
                else { append(text[i]); i++ }
            }
            text.startsWith("*", i) && !text.startsWith("**", i) -> {
                val end = text.indexOf("*", i + 1)
                if (end != -1 && !text.startsWith("*", end + 1)) {
                    withStyle(italic) { append(text.substring(i + 1, end)) }; i = end + 1
                } else { append(text[i]); i++ }
            }
            text.startsWith("~~", i) -> {
                val end = text.indexOf("~~", i + 2)
                if (end != -1) { withStyle(strike) { append(text.substring(i + 2, end)) }; i = end + 2 }
                else { append(text[i]); i++ }
            }
            text.startsWith("`", i) -> {
                val end = text.indexOf("`", i + 1)
                if (end != -1) { withStyle(code) { append(" ${text.substring(i + 1, end)} ") }; i = end + 1 }
                else { append(text[i]); i++ }
            }
            else -> { append(text[i]); i++ }
        }
    }
}

// ── Pulsing status dot ────────────────────────────────────────────────────────
@Composable
fun PulsingDot(color: Color, size: Float = 8f) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "alpha"
    )
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.85f, targetValue = 1.15f,
        animationSpec = infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "scale"
    )
    Box(
        Modifier
            .size((size * scale).dp)
            .background(color.copy(alpha = alpha), CircleShape)
    )
}

// ── Typing dots ───────────────────────────────────────────────────────────────
@Composable
fun TypingDots() {
    val transition = rememberInfiniteTransition(label = "dots")
    Row(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
        (0..2).forEach { idx ->
            val scale by transition.animateFloat(
                0.5f, 1f,
                infiniteRepeatable(
                    tween(600, delayMillis = idx * 160, easing = FastOutSlowInEasing),
                    RepeatMode.Reverse
                ),
                label = "dot$idx"
            )
            Box(
                Modifier
                    .size((5.5f * scale).dp)
                    .background(AccentTeal.copy(alpha = 0.5f + 0.5f * scale), CircleShape)
            )
        }
    }
}

// ── Top header bar ────────────────────────────────────────────────────────────
@Composable
fun VisionQAHeader(isReady: Boolean) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(BgDeep)
            .drawBehind {
                // Bottom border line
                drawLine(
                    color = DividerColor,
                    start = Offset(0f, size.height),
                    end = Offset(size.width, size.height),
                    strokeWidth = 1.dp.toPx()
                )
            }
            .padding(horizontal = 18.dp, vertical = 14.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Logo + name
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                // Icon badge
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .background(
                            Brush.linearGradient(listOf(AccentTeal, AccentIndigo)),
                            RoundedCornerShape(10.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "V",
                        style = TextStyle(
                            fontSize = 16.sp,
                            fontWeight = FontWeight.ExtraBold,
                            color = Color.White
                        )
                    )
                }
                Column {
                    Text(
                        text = "VisionQA",
                        style = TextStyle(
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary,
                            letterSpacing = (-0.3).sp
                        )
                    )
                    Text(
                        text = "Visual Intelligence",
                        style = TextStyle(
                            fontSize = 10.sp,
                            color = TextSecond,
                            letterSpacing = 0.4.sp
                        )
                    )
                }
            }

            // Status indicator
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier
                    .background(BgCard, RoundedCornerShape(20.dp))
                    .padding(horizontal = 10.dp, vertical = 5.dp)
            ) {
                if (isReady) {
                    PulsingDot(color = AccentTeal, size = 7f)
                    Text(
                        text = "Ready",
                        style = TextStyle(fontSize = 11.sp, color = AccentTeal, fontWeight = FontWeight.Medium)
                    )
                } else {
                    PulsingDot(color = AccentAmber, size = 7f)
                    Text(
                        text = "Loading",
                        style = TextStyle(fontSize = 11.sp, color = AccentAmber, fontWeight = FontWeight.Medium)
                    )
                }
            }
        }
    }
}

// ── Chat bubble ───────────────────────────────────────────────────────────────
@Composable
fun ChatBubble(
    message: ChatMessage,
    loadingPhase: String,
    onSpeakClick: (String) -> Unit
) {
    val isUser = message.isUser

    AnimatedVisibility(
        visible = true,
        enter = fadeIn(tween(200)) + slideInVertically(tween(200)) { it / 4 }
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp, horizontal = 14.dp),
            horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
        ) {
            if (message.isLoading) {
                Row(
                    modifier = Modifier
                        .background(AiBubble, RoundedCornerShape(6.dp, 18.dp, 18.dp, 18.dp))
                        .border(1.dp, DividerColor, RoundedCornerShape(6.dp, 18.dp, 18.dp, 18.dp))
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    TypingDots()
                    Text(
                        text = loadingPhase,
                        style = TextStyle(fontSize = 12.sp, color = TextSecond)
                    )
                }
            } else {
                // Sender label
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(5.dp),
                    modifier = Modifier.padding(
                        start = if (isUser) 0.dp else 4.dp,
                        end = if (isUser) 4.dp else 0.dp,
                        bottom = 4.dp
                    )
                ) {
                    if (!isUser) {
                        Box(
                            modifier = Modifier
                                .size(18.dp)
                                .background(
                                    Brush.linearGradient(listOf(AccentTeal, AccentIndigo)),
                                    CircleShape
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("V", style = TextStyle(fontSize = 9.sp, fontWeight = FontWeight.Bold, color = Color.White))
                        }
                    }
                    Text(
                        text = if (isUser) "You" else "VisionQA",
                        style = TextStyle(
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = if (isUser) AccentIndigo.copy(alpha = 0.85f) else AccentTeal.copy(alpha = 0.85f),
                            letterSpacing = 0.3.sp
                        )
                    )
                }

                Box(
                    modifier = Modifier
                        .let {
                            if (isUser) it.fillMaxWidth(0.84f) else it.fillMaxWidth(0.94f)
                        }
                        .shadow(
                            elevation = if (!isUser) 4.dp else 2.dp,
                            shape = if (isUser) RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp)
                            else RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp),
                            ambientColor = if (!isUser) AccentTeal.copy(alpha = 0.06f) else Color.Transparent,
                            spotColor = if (!isUser) AccentTeal.copy(alpha = 0.06f) else Color.Transparent
                        )
                        .background(
                            if (isUser) UserBubble else AiBubble,
                            if (isUser) RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp)
                            else RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp)
                        )
                        .border(
                            width = 1.dp,
                            color = if (isUser) AccentIndigo.copy(alpha = 0.2f) else DividerColor,
                            shape = if (isUser) RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp)
                            else RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp)
                        )
                        .drawBehind {
                            if (!isUser) {
                                // Left accent bar — teal to indigo gradient
                                drawRect(
                                    brush = Brush.verticalGradient(listOf(AccentTeal, AccentIndigo)),
                                    topLeft = Offset(0f, 0f),
                                    size = size.copy(width = 3.dp.toPx())
                                )
                            }
                        }
                        .padding(start = if (!isUser) 10.dp else 0.dp)
                ) {
                    Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
                        // Image thumbnail
                        message.image?.let { bmp ->
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(190.dp)
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(BgCard)
                            ) {
                                Image(
                                    bitmap = bmp.asImageBitmap(),
                                    contentDescription = "Captured image",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.Crop
                                )
                                // Gradient overlay
                                Box(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .background(
                                            Brush.verticalGradient(
                                                listOf(Color.Transparent, Color.Black.copy(alpha = 0.5f))
                                            )
                                        )
                                )
                                // "Analyzing..." badge shown on first image
                                Box(
                                    modifier = Modifier
                                        .align(Alignment.BottomStart)
                                        .padding(10.dp)
                                        .background(
                                            Color.Black.copy(alpha = 0.6f),
                                            RoundedCornerShape(6.dp)
                                        )
                                        .padding(horizontal = 8.dp, vertical = 4.dp)
                                ) {
                                    Text(
                                        "Image captured",
                                        style = TextStyle(fontSize = 10.sp, color = AccentTeal, fontWeight = FontWeight.Medium)
                                    )
                                }
                            }
                            Spacer(Modifier.height(10.dp))
                        }

                        // Text content
                        if (message.text.isNotEmpty()) {
                            if (isUser) {
                                Text(
                                    text = message.text,
                                    style = TextStyle(
                                        fontSize = 14.sp,
                                        lineHeight = 21.sp,
                                        color = TextPrimary
                                    )
                                )
                            } else {
                                MarkdownText(text = message.text, baseColor = TextPrimary)
                            }
                        }

                        // Speak button for AI messages
                        if (!isUser && message.text.isNotEmpty()) {
                            Spacer(Modifier.height(10.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End
                            ) {
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .clickable { onSpeakClick(message.text) }
                                        .background(AccentTeal.copy(alpha = 0.10f))
                                        .padding(horizontal = 10.dp, vertical = 5.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.VolumeUp,
                                            contentDescription = "Read aloud",
                                            tint = AccentTeal.copy(alpha = 0.8f),
                                            modifier = Modifier.size(14.dp)
                                        )
                                        Text(
                                            "Read aloud",
                                            style = TextStyle(fontSize = 11.sp, color = AccentTeal.copy(alpha = 0.8f))
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Download / init progress overlay ─────────────────────────────────────────
@Composable
fun DownloadProgressBar(message: String, percent: Float) {
    AnimatedVisibility(
        visible = message.isNotEmpty(),
        enter = expandVertically() + fadeIn(),
        exit = shrinkVertically() + fadeOut()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(BgCard)
                .padding(horizontal = 20.dp, vertical = 12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = message,
                    style = TextStyle(fontSize = 12.sp, color = TextSecond),
                    modifier = Modifier.weight(1f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "${(percent * 100).toInt()}%",
                    style = TextStyle(
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = AccentTeal
                    )
                )
            }
            Spacer(Modifier.height(8.dp))
            // Custom progress bar with animated fill
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(BgInput)
            ) {
                val animatedProgress by animateFloatAsState(
                    targetValue = percent,
                    animationSpec = tween(400),
                    label = "progress"
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth(animatedProgress.coerceIn(0f, 1f))
                        .fillMaxHeight()
                        .background(
                            Brush.horizontalGradient(listOf(AccentTeal, AccentIndigo)),
                            RoundedCornerShape(2.dp)
                        )
                )
            }
        }
    }
}

// ── Empty state (no image captured yet) ──────────────────────────────────────
@Composable
fun EmptyStateHint() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 12.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            modifier = Modifier
                .background(BgCard, RoundedCornerShape(12.dp))
                .border(1.dp, DividerColor, RoundedCornerShape(12.dp))
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                imageVector = Icons.Default.CameraAlt,
                contentDescription = null,
                tint = AccentTeal.copy(alpha = 0.6f),
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = "Tap Capture to take a photo, then ask anything about it",
                style = TextStyle(
                    fontSize = 12.sp,
                    color = TextSecond,
                    lineHeight = 18.sp
                )
            )
        }
    }
}

// ── Main screen ───────────────────────────────────────────────────────────────
@Composable
fun VisionQAScreen() {
    val context       = LocalContext.current
    val scope         = rememberCoroutineScope()

    val vlm           = remember { GemmaVLM(context) }
    val ocrHelper     = remember { OcrHelper() }
    val speechManager = remember { SpeechManager(context) }

    var imageCapture     by remember { mutableStateOf<ImageCapture?>(null) }
    var capturedBitmap   by remember { mutableStateOf<Bitmap?>(null) }
    var cachedOcrResult  by remember { mutableStateOf<OcrHelper.OcrResult?>(null) }

    var questionText     by remember { mutableStateOf("") }
    var messages         by remember { mutableStateOf(listOf<ChatMessage>()) }
    var isProcessing     by remember { mutableStateOf(false) }
    var isInitializing   by remember { mutableStateOf(true) }
    var isListening      by remember { mutableStateOf(false) }
    var downloadProgress by remember { mutableStateOf("") }
    var downloadPercent  by remember { mutableFloatStateOf(0f) }
    var inferenceJob     by remember { mutableStateOf<Job?>(null) }
    var loadingPhase     by remember { mutableStateOf("Thinking...") }

    // Whether the model is fully loaded and usable
    val isReady = !isInitializing && messages.any { !it.isUser && it.text.startsWith("Ready") }

    val listState = rememberLazyListState()

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            isListening = true
            scope.launch {
                speechManager.startListening().collect { text -> questionText = text }
                isListening = false
            }
        }
    }

    LaunchedEffect(messages.size, messages.lastOrNull()?.text?.length) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
    }

    if (isProcessing) {
        LaunchedEffect(Unit) {
            val phases = listOf("Analyzing image...", "Reasoning...", "Generating answer...", "Almost done...")
            var i = 0
            while (true) { loadingPhase = phases[i++ % phases.size]; delay(2800) }
        }
    }

    // ── Initialization (replace "Gemma" mentions with "VisionQA") ────────────
    LaunchedEffect(Unit) {
        messages = listOf(ChatMessage(text = "Initializing VisionQA...", isUser = false))
        scope.launch {
            try {
                val ok = vlm.initialize { msg, prog ->
                    // Strip model names from progress messages shown to user
                    val sanitized = msg
                        .replace(Regex("(?i)gemma[\\s\\w]*"), "VisionQA")
                        .replace(Regex("(?i)brain|eyes|projector|mmproj", RegexOption.IGNORE_CASE), "model data")
                        .replace("model engine", "VisionQA engine")
                    downloadProgress = sanitized
                    downloadPercent  = prog / 100f
                }
                messages = if (ok)
                    listOf(ChatMessage(text = "Ready! Capture an image to get started.", isUser = false))
                else
                    listOf(ChatMessage(text = "Setup failed. Please check your connection and restart.", isUser = false))
                isInitializing = false
                downloadProgress = ""
            } catch (e: Exception) {
                messages = listOf(ChatMessage(text = "Something went wrong. Please restart the app.", isUser = false))
                isInitializing = false
            }
        }
    }

    val view = androidx.compose.ui.platform.LocalView.current
    DisposableEffect(isInitializing) {
        view.keepScreenOn = isInitializing
        onDispose { view.keepScreenOn = false }
    }

    // ── Root layout ───────────────────────────────────────────────────────────
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {

            // ── Header ────────────────────────────────────────────────────────
            VisionQAHeader(isReady = !isInitializing)

            // ── Camera / image panel ──────────────────────────────────────────
            Box(
                modifier = Modifier
                    .weight(0.38f)
                    .fillMaxWidth()
            ) {
                if (capturedBitmap != null) {
                    Image(
                        bitmap = capturedBitmap!!.asImageBitmap(),
                        contentDescription = "Captured",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                    // Subtle top fade to blend with header
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(40.dp)
                            .background(
                                Brush.verticalGradient(listOf(BgDeep.copy(alpha = 0.6f), Color.Transparent))
                            )
                    )
                } else {
                    CameraPreview(
                        modifier = Modifier.fillMaxSize(),
                        onImageCaptureReady = { imageCapture = it }
                    )
                    // Overlay corner guides when camera active
                    Box(modifier = Modifier.fillMaxSize()) {
                        // Corner accent decorations
                        listOf(
                            Alignment.TopStart, Alignment.TopEnd,
                            Alignment.BottomStart, Alignment.BottomEnd
                        ).forEach { alignment ->
                            Box(
                                modifier = Modifier
                                    .align(alignment)
                                    .padding(16.dp)
                                    .size(24.dp)
                            ) {
                                // Could add corner bracket SVG here — placeholder box
                                Box(
                                    Modifier
                                        .size(16.dp)
                                        .border(
                                            width = 2.dp,
                                            color = AccentTeal.copy(alpha = 0.6f),
                                            shape = when (alignment) {
                                                Alignment.TopStart -> RoundedCornerShape(topStart = 4.dp)
                                                Alignment.TopEnd -> RoundedCornerShape(topEnd = 4.dp)
                                                Alignment.BottomStart -> RoundedCornerShape(bottomStart = 4.dp)
                                                else -> RoundedCornerShape(bottomEnd = 4.dp)
                                            }
                                        )
                                )
                            }
                        }
                    }
                }
            }

            // ── Download progress ─────────────────────────────────────────────
            DownloadProgressBar(message = downloadProgress, percent = downloadPercent)

            // ── Messages ──────────────────────────────────────────────────────
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(0.62f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(top = 10.dp, bottom = 8.dp)
            ) {
                items(messages) { msg ->
                    ChatBubble(
                        message = msg,
                        loadingPhase = loadingPhase,
                        onSpeakClick = { speechManager.speak(it) }
                    )
                    Spacer(Modifier.height(2.dp))
                }
                // Show hint when no image is captured and we're ready
                if (capturedBitmap == null && !isInitializing && messages.size <= 1) {
                    item {
                        EmptyStateHint()
                    }
                }
            }

            // ── Input panel ───────────────────────────────────────────────────
            Surface(
                color = BgCard,
                tonalElevation = 0.dp,
                modifier = Modifier
                    .fillMaxWidth()
                    .drawBehind {
                        drawLine(
                            color = DividerColor,
                            start = Offset(0f, 0f),
                            end = Offset(size.width, 0f),
                            strokeWidth = 1.dp.toPx()
                        )
                    }
            ) {
                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {

                    // ── Image not captured — show Capture button prominently ──
                    if (capturedBitmap == null) {
                        Button(
                            onClick = {
                                val capture = imageCapture ?: return@Button
                                capture.takePicture(
                                    ContextCompat.getMainExecutor(context),
                                    object : ImageCapture.OnImageCapturedCallback() {
                                        override fun onCaptureSuccess(image: ImageProxy) {
                                            try {
                                                val bmp = imageProxyToBitmap(image)
                                                vlm.resetForNewImage()
                                                capturedBitmap = bmp
                                                scope.launch {
                                                    cachedOcrResult = try {
                                                        ocrHelper.extractText(bmp)
                                                    } catch (_: Exception) { null }
                                                }
                                            } finally {
                                                image.close()
                                            }
                                        }
                                        override fun onError(exc: ImageCaptureException) {
                                            messages = messages + ChatMessage(
                                                text = "Camera error. Please try again.", isUser = false
                                            )
                                        }
                                    }
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(48.dp),
                            enabled = !isInitializing,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = AccentTeal,
                                contentColor = Color(0xFF0F1117),
                                disabledContainerColor = BgInput,
                                disabledContentColor = TextSecond
                            ),
                            shape = RoundedCornerShape(14.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.CameraAlt,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = if (isInitializing) "Initializing VisionQA..." else "Capture Image",
                                style = TextStyle(fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                            )
                        }
                    } else {
                        // ── Image captured — show text input + controls ───────
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            // Mic button
                            IconButton(
                                onClick = {
                                    if (isListening) isListening = false
                                    else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                },
                                modifier = Modifier
                                    .size(44.dp)
                                    .background(
                                        if (isListening) AccentTeal.copy(alpha = 0.2f) else BgInput,
                                        CircleShape
                                    )
                                    .border(
                                        1.dp,
                                        if (isListening) AccentTeal.copy(alpha = 0.5f) else Color.Transparent,
                                        CircleShape
                                    )
                            ) {
                                Icon(
                                    imageVector = if (isListening) Icons.Default.Stop else Icons.Default.Mic,
                                    contentDescription = "Voice input",
                                    tint = if (isListening) AccentTeal else TextSecond,
                                    modifier = Modifier.size(20.dp)
                                )
                            }

                            // Text field
                            BasicTextField(
                                value = questionText,
                                onValueChange = { questionText = it },
                                enabled = !isProcessing && !isInitializing,
                                modifier = Modifier
                                    .weight(1f)
                                    .background(BgInput, RoundedCornerShape(22.dp))
                                    .border(
                                        1.dp,
                                        if (questionText.isNotEmpty()) AccentTeal.copy(alpha = 0.3f) else Color.Transparent,
                                        RoundedCornerShape(22.dp)
                                    )
                                    .padding(horizontal = 16.dp, vertical = 11.dp),
                                textStyle = TextStyle(
                                    fontSize = 14.sp,
                                    color = TextPrimary,
                                    lineHeight = 20.sp
                                ),
                                maxLines = 4,
                                decorationBox = { inner ->
                                    if (questionText.isEmpty()) {
                                        Text(
                                            "Ask about the image...",
                                            style = TextStyle(fontSize = 14.sp, color = TextSecond.copy(alpha = 0.7f))
                                        )
                                    }
                                    inner()
                                }
                            )

                            // Send / Stop
                            if (isProcessing) {
                                IconButton(
                                    onClick = {
                                        inferenceJob?.cancel()
                                        vlm.abort() // Immediately kill the C++/MediaPipe generation engine
                                        isProcessing = false
                                        messages = messages.mapIndexed { index, msg ->
                                            // Target the active AI message (which is always the last one)
                                            if (index == messages.lastIndex && !msg.isUser) {
                                                msg.copy(
                                                    isLoading = false,
                                                    text = if (msg.text.isEmpty()) "Stopped." else msg.text + " 🛑 [Stopped]"
                                                )
                                            } else msg
                                        }
                                    },
                                    modifier = Modifier
                                        .size(44.dp)
                                        .background(Color(0xFF3D1515), CircleShape)
                                        .border(1.dp, Color(0xFFFF4444).copy(alpha = 0.4f), CircleShape)
                                ) {
                                    Icon(
                                        Icons.Default.Stop, "Stop",
                                        tint = Color(0xFFFF6B6B),
                                        modifier = Modifier.size(20.dp)
                                    )
                                }
                            }

                            else {
                                val canSend = capturedBitmap != null && questionText.isNotBlank()
                                IconButton(
                                    onClick = {
                                        val bitmap   = capturedBitmap ?: return@IconButton
                                        val question = questionText.trim()
                                        if (question.isBlank()) return@IconButton
                                        questionText = ""
                                        isProcessing = true

                                        val isFirst = messages.none { !it.isUser }
                                        messages = messages + ChatMessage(
                                            text = question, isUser = true,
                                            image = if (isFirst) bitmap else null
                                        )
                                        val aiId = UUID.randomUUID().toString()
                                        messages = messages + ChatMessage(
                                            id = aiId, text = "", isUser = false, isLoading = true
                                        )

                                        inferenceJob = scope.launch {
                                            try {
                                                val answer = vlm.chat(
                                                    image = bitmap,
                                                    question = question,
                                                    onTokenGenerated = { token ->
                                                        if (isProcessing) { // <--- Ignore late callbacks if generation was stopped
                                                            messages = messages.map {
                                                                if (it.id == aiId)
                                                                    it.copy(isLoading = false, text = it.text + token)
                                                                else it
                                                            }
                                                        }
                                                    }
                                                )

                                                messages = messages.map {
                                                    if (it.id == aiId && it.text.isEmpty())
                                                        it.copy(isLoading = false, text = answer)
                                                    else it
                                                }
                                            } catch (_: CancellationException) {
                                            } catch (e: Exception) {
                                                messages = messages.map {
                                                    if (it.id == aiId)
                                                        it.copy(isLoading = false, text = "Something went wrong. Please try again.")
                                                    else it
                                                }
                                            } finally {
                                                isProcessing = false
                                            }
                                        }
                                    },
                                    enabled = canSend,
                                    modifier = Modifier
                                        .size(44.dp)
                                        .background(
                                            if (canSend) AccentTeal else BgInput,
                                            CircleShape
                                        )
                                ) {
                                    Icon(
                                        Icons.Default.Send, "Send",
                                        tint = if (canSend) Color(0xFF0F1117) else TextSecond,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }

                        Spacer(Modifier.height(8.dp))

                        // Retake button (compact, secondary)
                        OutlinedButton(
                            onClick = {
                                vlm.resetForNewImage()
                                capturedBitmap  = null
                                cachedOcrResult = null
                                messages        = emptyList()
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(40.dp),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = TextSecond
                            ),
                            border = androidx.compose.foundation.BorderStroke(1.dp, DividerColor),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(6.dp))
                            Text(
                                text = "New Image",
                                style = TextStyle(fontSize = 13.sp, fontWeight = FontWeight.Medium)
                            )
                        }
                    }
                }
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            vlm.cleanup()
            ocrHelper.close()
            speechManager.cleanup()
        }
    }
}

// ── BasicTextField alias ──────────────────────────────────────────────────────
@Composable
private fun BasicTextField(
    value: String,
    onValueChange: (String) -> Unit,
    enabled: Boolean,
    modifier: Modifier,
    textStyle: TextStyle,
    maxLines: Int,
    decorationBox: @Composable (innerTextField: @Composable () -> Unit) -> Unit
) {
    androidx.compose.foundation.text.BasicTextField(
        value = value,
        onValueChange = onValueChange,
        enabled = enabled,
        modifier = modifier,
        textStyle = textStyle,
        maxLines = maxLines,
        decorationBox = decorationBox
    )
}