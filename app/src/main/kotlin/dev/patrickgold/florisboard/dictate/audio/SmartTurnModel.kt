/*
 * Copyright (C) 2026 DevEmperor (Dictate)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

package dev.patrickgold.florisboard.dictate.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import dev.patrickgold.florisboard.dictate.provider.LocalModelCatalog
import dev.patrickgold.florisboard.dictate.provider.LocalTranscriptionProvider
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Process-wide Smart Turn v3.2 CPU session. The first prediction initializes the model; subsequent
 * recordings reuse the optimized session so auto-segmentation does not repeatedly pay model-load cost.
 *
 * The bundled model is the official Pipecat quantized CPU classifier with Pipecat's Whisper feature
 * extractor prepended by `tools/build-smart-turn-model.py`. It accepts exactly eight seconds of 16 kHz
 * mono float PCM, left-padded with zeroes as required by Smart Turn.
 */
internal object SmartTurnModel {
    const val SAMPLE_COUNT = AudioDecode.TARGET_SAMPLE_RATE * 8

    /** Fixed on-disk name inside the model dir; the file is downloaded on demand, not bundled. */
    private const val MODEL_DEST = "smart-turn.onnx"
    private const val MODEL_BYTES = 8_840_701L
    private const val COMPLETE_THRESHOLD = 0.5f
    private const val QNN_HEXAGON_BACKEND_PATH = "libQnnHtp.so"

    @Volatile private var holder: SessionHolder? = null
    // Set after a native session-creation failure so a broken runtime is not retried on every pause. The
    // model file merely being absent (not yet downloaded) does NOT set this — that check is cheap.
    @Volatile private var unavailable = false
    private val lock = Any()

    /** The on-disk model file (downloaded via the Smart Turn checkbox); may not exist yet. */
    private fun modelFile(context: Context): File =
        File(LocalTranscriptionProvider.modelDir(context, LocalModelCatalog.SMART_TURN_ID), MODEL_DEST)

    /** True when the downloaded model is present and intact — gates activation and the settings checkbox. */
    fun isModelAvailable(context: Context): Boolean {
        val f = modelFile(context.applicationContext)
        return f.isFile && f.length() == MODEL_BYTES
    }

    /** Returns null when the local model/runtime cannot be used, allowing the silence fallback to win. */
    fun predictsComplete(context: Context, audio: FloatArray): Boolean? {
        if (audio.size != SAMPLE_COUNT || unavailable) return null
        val active = holder ?: synchronized(lock) {
            holder ?: createSession(context.applicationContext)?.also { holder = it }
        } ?: return null
        return synchronized(active) { active.predict(audio) }
    }

    private fun createSession(context: Context): SessionHolder? {
        // Absent model = not downloaded yet → fall back quietly, but keep retrying cheaply once it lands.
        val model = modelFile(context).takeIf { it.isFile && it.length() == MODEL_BYTES } ?: return null
        return runCatching {
            val environment = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setInterOpNumThreads(1)
                setIntraOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                configureLocalAcceleration()
            }
            try {
                SessionHolder(environment, environment.createSession(model.absolutePath, options))
            } finally {
                options.close()
            }
        }.getOrElse {
            // A broken native runtime (e.g. an ORT/JNI mismatch) or corrupt model must not be retried on
            // every pause — give up Smart Turn for this process; the silence fallback keeps working.
            unavailable = true
            null
        }
    }

    private class SessionHolder(
        private val environment: OrtEnvironment,
        private val session: OrtSession,
    ) {
        private val inputBuffer = ByteBuffer.allocateDirect(SAMPLE_COUNT * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        fun predict(audio: FloatArray): Boolean? = runCatching {
            inputBuffer.clear()
            inputBuffer.put(audio)
            inputBuffer.rewind()
            OnnxTensor.createTensor(environment, inputBuffer, longArrayOf(1, SAMPLE_COUNT.toLong())).use { input ->
                session.run(mapOf("audio" to input)).use { result ->
                    val probability = (result.get(0) as OnnxTensor).floatBuffer.get(0)
                    probability > COMPLETE_THRESHOLD
                }
            }
        }.getOrNull()
    }

    /**
     * Prefers Qualcomm's QNN Hexagon path on likely Snapdragon devices and falls back to NNAPI, then CPU.
     * All provider enablement is best-effort: unsupported execution providers simply keep the CPU path.
     */
    private fun OrtSession.SessionOptions.configureLocalAcceleration() {
        var qnnEnabled = false
        if (isLikelySnapdragon()) {
            qnnEnabled = try {
                addQnn(mapOf("backend_path" to QNN_HEXAGON_BACKEND_PATH))
                true
            } catch (_: OrtException) {
                false
            }
        }
        if (!qnnEnabled && Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                addNnapi()
            } catch (_: OrtException) {
                // Best effort only; CPU remains available.
            }
        }
    }

    private fun isLikelySnapdragon(): Boolean {
        val probes = buildList {
            add(Build.HARDWARE.orEmpty())
            add(Build.BOARD.orEmpty())
            add(Build.PRODUCT.orEmpty())
            add(Build.BRAND.orEmpty())
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                add(Build.SOC_MANUFACTURER.orEmpty())
                add(Build.SOC_MODEL.orEmpty())
            }
        }
        return probes.any { value ->
            val normalized = value.lowercase()
            normalized.contains("qualcomm") ||
                normalized.contains("snapdragon") ||
                normalized.contains("qcom")
        }
    }
}

/** Fixed-size rolling turn buffer matching Pipecat's last-eight-seconds + left-padding contract. */
internal class SmartTurnPcmBuffer(private val capacity: Int = SmartTurnModel.SAMPLE_COUNT) {
    private val samples = ShortArray(capacity)
    private var writeIndex = 0
    private var size = 0

    fun append(chunk: ShortArray) {
        if (chunk.size >= capacity) {
            chunk.copyInto(samples, startIndex = chunk.size - capacity)
            writeIndex = 0
            size = capacity
            return
        }
        chunk.forEach { sample ->
            samples[writeIndex] = sample
            writeIndex = (writeIndex + 1) % capacity
            if (size < capacity) size++
        }
    }

    fun clear() {
        writeIndex = 0
        size = 0
    }

    /** Drops old pre-roll while preserving the newest [count] samples in their existing ring positions. */
    fun keepNewest(count: Int) {
        size = size.coerceAtMost(count.coerceIn(0, capacity))
    }

    fun snapshotNormalizedLeftPadded(): FloatArray {
        val output = FloatArray(capacity)
        val destinationStart = capacity - size
        var source = (writeIndex - size + capacity) % capacity
        for (i in 0 until size) {
            output[destinationStart + i] = samples[source] / 32768f
            source = (source + 1) % capacity
        }
        return output
    }
}
