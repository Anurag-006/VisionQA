package com.anurag.visionqa.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import java.util.Locale

class SpeechManager(context: Context) {

    private var speechRecognizer: SpeechRecognizer? = SpeechRecognizer.createSpeechRecognizer(context)
    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("SpeechManager", "TTS Language not supported")
                } else {
                    isTtsReady = true
                }
            } else {
                Log.e("SpeechManager", "TTS Initialization failed")
            }
        }
    }

    // --- TEXT TO SPEECH (OUTPUT) ---
    fun speak(text: String) {
        if (isTtsReady) {
            // Stop any previous speech
            tts?.stop()
            // Queue the new text
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    fun stopSpeaking() {
        tts?.stop()
    }

    // --- SPEECH TO TEXT (INPUT) ---
    fun startListening(): Flow<String> = callbackFlow {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }

        val listener = object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                // close() // Don't close immediately on error to allow retry logic if needed
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    trySend(matches[0]) // Send the final recognized text
                }
                close()
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    trySend(matches[0]) // Send partial updates (live typing effect)
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }

        speechRecognizer?.setRecognitionListener(listener)
        speechRecognizer?.startListening(intent)

        awaitClose {
            // Cleanup when the flow is cancelled
            // We don't stop the recognizer here to avoid cutting off too early in some flows
        }
    }

    fun cleanup() {
        speechRecognizer?.destroy()
        tts?.stop()
        tts?.shutdown()
    }
}