/*
 * Copyright (C) 2026 DevEmperor (Dictate)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

package dev.patrickgold.florisboard.dictate.recognition

import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.RecognitionService
import android.speech.SpeechRecognizer

/**
 * Makes Dictate a system-wide speech recognizer via the standard Android [RecognitionService] API
 * (issue #67) — the same mechanism FUTO Voice Input uses. Once selected as the device's recognition
 * service, any app/keyboard that speaks the standard voice-input protocol can dictate through Dictate's
 * engine (local sherpa/Whisper or a cloud provider) and insert the result itself.
 *
 * All the actual work (recording, transcription, endpointing) lives in the shared [RecognitionSession];
 * this class only bridges its [RecognitionSession.Host] callbacks to the caller's [Callback].
 *
 * Note: some keyboards (notably Gboard) hardwire their mic to Google and won't use a third-party
 * recognizer — this works with apps/keyboards that honour the system default recognition service. The
 * `ACTION_RECOGNIZE_SPEECH` popup path is handled separately by [RecognitionActivity].
 */
class DictateRecognitionService : RecognitionService() {

    private var session: RecognitionSession? = null

    override fun onStartListening(recognizerIntent: Intent, listener: Callback) {
        session?.cancel()
        runCatching { listener.readyForSpeech(Bundle()) }
        session = RecognitionSession(
            appContext = applicationContext,
            host = hostFor(listener),
            endpointing = endpointingFor(recognizerIntent),
        ).also { it.start() }
    }

    override fun onStopListening(listener: Callback) {
        session?.stop()
    }

    override fun onCancel(listener: Callback) {
        session?.cancel()
        session = null
    }

    override fun onDestroy() {
        session?.cancel()
        session = null
        super.onDestroy()
    }

    private fun hostFor(listener: Callback) = object : RecognitionSession.Host {
        override fun onBeginningOfSpeech() {
            runCatching { listener.beginningOfSpeech() }
        }

        override fun onEndOfSpeech() {
            runCatching { listener.endOfSpeech() }
        }

        override fun onRmsChanged(rmsdB: Float) {
            runCatching { listener.rmsChanged(rmsdB) }
        }

        override fun onPartial(text: String) {
            runCatching { listener.partialResults(resultsBundle(text)) }
        }

        override fun onResults(text: String) {
            runCatching { listener.results(resultsBundle(text)) }
            session = null
        }

        override fun onError(code: Int) {
            runCatching { listener.error(code) }
            session = null
        }
    }

    private companion object {
        private const val MIN_END_SILENCE_MS = 500L
        private const val MAX_END_SILENCE_MS = 10_000L
        private const val MIN_NO_SPEECH_TIMEOUT_MS = 1_000L
        private const val MAX_NO_SPEECH_TIMEOUT_MS = 30_000L
        private const val MIN_MINIMUM_LENGTH_MS = 0L
        private const val MAX_MINIMUM_LENGTH_MS = 10 * 60 * 1_000L
        private const val MIN_MAX_RECORDING_MS = 5_000L
        private const val MAX_MAX_RECORDING_MS = 10 * 60 * 1_000L
        private const val EXTRA_DICTATE_NO_SPEECH_TIMEOUT_MS =
            "net.devemperor.dictate.extra.NO_SPEECH_TIMEOUT_MS"
        private const val EXTRA_DICTATE_MAX_RECORDING_MS =
            "net.devemperor.dictate.extra.MAX_RECORDING_MS"

        private fun endpointingFor(intent: Intent): RecognitionSession.EndpointingConfig {
            val defaults = RecognitionSession.EndpointingConfig.defaultForDevice()
            val completeSilenceMs = intent.getLongExtra(
                RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS,
                defaults.endSilenceMs,
            ).coerceIn(MIN_END_SILENCE_MS, MAX_END_SILENCE_MS)
            val possibleSilenceMs = intent.getLongExtra(
                RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS,
                completeSilenceMs,
            ).coerceIn(MIN_END_SILENCE_MS, MAX_END_SILENCE_MS)
            val endSilenceMs = when {
                intent.hasExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS) ->
                    completeSilenceMs
                intent.hasExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS) ->
                    possibleSilenceMs
                else -> defaults.endSilenceMs
            }
            val minimumLengthHintMs = if (intent.hasExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS,
                )
            ) {
                intent.getLongExtra(
                    RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS,
                    0L,
                ).coerceIn(MIN_MINIMUM_LENGTH_MS, MAX_MINIMUM_LENGTH_MS)
            } else {
                null
            }
            val noSpeechTimeoutMs = intent.getLongExtra(
                EXTRA_DICTATE_NO_SPEECH_TIMEOUT_MS,
                defaults.noSpeechTimeoutMs,
            ).coerceIn(MIN_NO_SPEECH_TIMEOUT_MS, MAX_NO_SPEECH_TIMEOUT_MS)
            val maxRecordingMs = intent.getLongExtra(
                EXTRA_DICTATE_MAX_RECORDING_MS,
                defaults.maxRecordingMs,
            ).coerceIn(MIN_MAX_RECORDING_MS, MAX_MAX_RECORDING_MS)
            val minimumLengthMs = minOf(
                minimumLengthHintMs ?: defaults.minimumLengthMs,
                maxRecordingMs,
            )
            return RecognitionSession.EndpointingConfig(
                endSilenceMs = endSilenceMs,
                noSpeechTimeoutMs = noSpeechTimeoutMs,
                minimumLengthMs = minimumLengthMs,
                maxRecordingMs = maxRecordingMs,
            )
        }

        private fun resultsBundle(text: String): Bundle = Bundle().apply {
            putStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION, arrayListOf(text))
        }
    }
}
