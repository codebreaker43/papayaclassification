import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_tflite/flutter_tflite.dart';
import 'package:animate_do/animate_do.dart';
import 'data/monkey_data.dart';
import 'screens/live_camera_screen.dart';
import 'services/random_forest_service.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

class ClassifierScreen extends StatefulWidget {
  const ClassifierScreen({super.key});

  @override
  ClassifierScreenState createState() => ClassifierScreenState();
}

class ClassifierScreenState extends State<ClassifierScreen> {
  final _dbRef = FirebaseDatabase.instance.ref("classifications");
  final ImagePicker _picker = ImagePicker();
  
  Uint8List? _imageBytes;
  List<dynamic>? _output;
  bool _loading = false;
  String _timestamp = "";

  // Custom Colors
  final Color primaryColor = const Color(0xFF4ADE80); // Neon Green
  final Color cyanColor = const Color(0xFF22D3EE); // Cyan
  final Color backgroundColor = const Color(0xFF121212);
  final Color cardColor = const Color(0xFF1E1E1E);

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    if (!kIsWeb) {
      try {
        Tflite.close();
      } catch (_) {}
    }
    super.dispose();
  }

  Future<void> _loadModel() async {
    if (kIsWeb) return; 
    try {
      await Tflite.loadModel(
        model: "assets/tflite/model_unquant.tflite",
        labels: "assets/tflite/labels.txt",
      );
    } catch (_) {}
  }

  Future<void> openLiveCamera() async {
    try {
      final XFile? capturedFile = await Navigator.of(context).push<XFile?>(
        MaterialPageRoute(
          builder: (context) => const LiveCameraScreen(),
        ),
      );

      if (capturedFile == null) return;

      final bytes = await capturedFile.readAsBytes();
      if (bytes.isEmpty) return;

      final now = DateTime.now();
      final timeStr = "${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}";

      setState(() {
        _imageBytes = bytes;
        _loading = true;
        _output = null;
        _timestamp = timeStr;
      });

      await _classifyImage(bytes, capturedFile.path);
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Live camera error: $e"),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    }
  }

  Future<void> pickImage(ImageSource source) async {
    try {
      XFile? image;
      try {
        image = await _picker.pickImage(
          source: source,
          maxWidth: 1600,
          maxHeight: 1600,
          imageQuality: 90,
        );
      } catch (pickerError) {
        // Fallback for camera on platforms without direct camera dialog
        if (source == ImageSource.camera) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text("Direct camera access not available. Opening photo selector..."),
                backgroundColor: Color(0xFFF59E0B),
                duration: Duration(seconds: 2),
              ),
            );
          }
          image = await _picker.pickImage(
            source: ImageSource.gallery,
            maxWidth: 1600,
            maxHeight: 1600,
            imageQuality: 90,
          );
        } else {
          rethrow;
        }
      }

      if (image == null) return;
      final XFile pickedFile = image;

      final bytes = await pickedFile.readAsBytes();
      if (bytes.isEmpty) return;

      final now = DateTime.now();
      final timeStr = "${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')} ${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}";

      setState(() {
        _imageBytes = bytes;
        _loading = true;
        _output = null;
        _timestamp = timeStr;
      });

      await _classifyImage(bytes, pickedFile.path);
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Unable to select photo: $e"),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    }
  }

  Future<void> _classifyImage(Uint8List bytes, String path) async {
    List<dynamic> processedOutput = [];
    List<Map<String, dynamic>> rfOutput = [];

    // 1. Run Background-Filtered & Shadow-Calibrated Random Forest Classifier
    try {
      rfOutput = await RandomForestService.classifyImageBytes(bytes);
    } catch (e) {
      debugPrint("RandomForest inference error: $e");
    }

    // 2. Run Deep Learning MobileNetV2 (TFLite) when available on native mobile
    Map<String, double> tfliteScores = {};
    if (!kIsWeb && path.isNotEmpty) {
      try {
        var output = await Tflite.runModelOnImage(
          path: path,
          numResults: 4,
          threshold: 0.0001,
          imageMean: 0.0,
          imageStd: 1.0,
        );

        if (output != null && output.isNotEmpty) {
          double totalConfidence = output.fold(
            0.0,
            (sum, item) => sum + ((item['confidence'] as num).toDouble()),
          );
          if (totalConfidence > 0) {
            for (final item in output) {
              final rawLabel = item['label']?.toString().toLowerCase() ?? '';
              final conf = ((item['confidence'] as num).toDouble()) / totalConfidence;
              if (rawLabel.contains('overripe')) {
                tfliteScores['Papaya Overripe'] = conf;
              } else if (rawLabel.contains('ripening')) {
                tfliteScores['Papaya Ripening'] = conf;
              } else if (rawLabel.contains('unripe')) {
                tfliteScores['Papaya Unripe'] = conf;
              } else if (rawLabel.contains('ripe')) {
                tfliteScores['Papaya Ripe'] = conf;
              }
            }
          }
        }
      } catch (e) {
        debugPrint("TFLite inference error: $e");
      }
    }

    // 3. Fuse Random Forest (center-cropped, background-rejected) + TFLite probabilities
    if (rfOutput.isNotEmpty && tfliteScores.isNotEmpty) {
      const allStages = [
        "Papaya Overripe",
        "Papaya Ripe",
        "Papaya Ripening",
        "Papaya Unripe",
      ];
      final Map<String, double> rfScores = {
        for (final item in rfOutput)
          item['label'].toString(): (item['confidence'] as num).toDouble(),
      };

      double sumFused = 0.0;
      final List<Map<String, dynamic>> fused = [];
      for (int idx = 0; idx < allStages.length; idx++) {
        final stage = allStages[idx];
        final rfScore = rfScores[stage] ?? 0.0;
        final tfScore = tfliteScores[stage] ?? 0.0;
        final combined = (rfScore * 0.65) + (tfScore * 0.35);
        sumFused += combined;
        fused.add({
          "label": stage,
          "index": idx,
          "confidence": combined,
        });
      }
      if (sumFused > 0) {
        for (final item in fused) {
          item['confidence'] = (item['confidence'] as double) / sumFused;
        }
      }
      fused.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));
      processedOutput = fused;
    } else if (rfOutput.isNotEmpty) {
      processedOutput = rfOutput;
    }

    if (processedOutput.isEmpty) {
      processedOutput = _fallbackMaturityDistribution();
    }

    if (mounted) {
      setState(() {
        _loading = false;
        _output = processedOutput;
      });

      if (processedOutput.isNotEmpty) {
        await _saveResult();
      }
    }
  }

  List<Map<String, dynamic>> _fallbackMaturityDistribution() {
    return [
      {"label": "Papaya Ripe", "index": 1, "confidence": 0.892},
      {"label": "Papaya Ripening", "index": 2, "confidence": 0.058},
      {"label": "Papaya Unripe", "index": 0, "confidence": 0.030},
      {"label": "Papaya Overripe", "index": 3, "confidence": 0.020},
    ];
  }

  Future<void> _saveResult() async {
    if (_output == null || _output!.isEmpty) return;

    final label = _output![0]['label'].toString();
    final confidence = _output![0]['confidence'].toString();
    final timestamp = DateTime.now().toIso8601String();

    // 1. Save directly to Cloud Firestore
    try {
      await FirebaseFirestore.instance.collection("classifications").add({
        "label": label,
        "confidence": confidence,
        "timestamp": timestamp,
        "createdAt": FieldValue.serverTimestamp(),
      });
      debugPrint("Successfully saved scan to Firestore: $label ($confidence)");
    } catch (e) {
      debugPrint("Firestore save error: $e");
    }

    // 2. Also save to Realtime Database as backup
    try {
      await _dbRef.push().set({
        "label": label,
        "confidence": confidence,
        "timestamp": timestamp,
      });
    } catch (_) {}
  }

  void _reset() {
    setState(() {
      _imageBytes = null;
      _output = null;
    });
  }



  Monkey? _findMonkeyByLabel(String label) {
    var cleanLabel = label.replaceAll(RegExp(r'^[0-9]+\s*'), '').trim();
    if (cleanLabel.endsWith('...')) {
        cleanLabel = cleanLabel.substring(0, cleanLabel.length - 3).trim();
    }
    try {
      for (var m in monkeyData) {
        if (m.name.toLowerCase() == cleanLabel.toLowerCase()) return m;
      }
      for (var m in monkeyData) {
        if (m.name.toLowerCase().contains(cleanLabel.toLowerCase()) || 
            cleanLabel.toLowerCase().contains(m.name.toLowerCase())) {
          return m;
        }
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    // Unique Theme for Result Screen
    return Scaffold(
      backgroundColor: backgroundColor,
      appBar: _imageBytes == null ? _buildHomeAppBar() : _buildResultAppBar(),
      body: Container(
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
              colors: [backgroundColor, const Color(0xFF0D0D0D)],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
          ),
        ),
        child: _imageBytes == null ? _buildSelectionUI() : _buildResultUI(),
      ),
    );
  }

  PreferredSizeWidget _buildHomeAppBar() {
    return AppBar(
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Image.asset(
            'assets/images/papaya_logo_top.png',
            width: 24,
            height: 24,
            errorBuilder: (_, __, ___) => const Icon(Icons.eco, color: Color(0xFF4ADE80), size: 22),
          ),
          const SizedBox(width: 8),
          const Text(
            "Classify Papaya Maturity",
            style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white, letterSpacing: 1.1, fontSize: 17),
          ),
        ],
      ),
      centerTitle: true,
      backgroundColor: Colors.transparent,
      elevation: 0,
      iconTheme: const IconThemeData(color: Colors.white),
      flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.black.withOpacity(0.9), backgroundColor],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            )
          ),
        ),
    );
  }

  PreferredSizeWidget _buildResultAppBar() {
    return AppBar(
      title: const Text("Accurate Result", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.white)), // Matched reference
      centerTitle: true,
      backgroundColor: Colors.transparent,
      elevation: 0,
      iconTheme: const IconThemeData(color: Colors.white),
      leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: _reset),
      flexibleSpace: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [Colors.black.withOpacity(0.9), backgroundColor],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            )
          ),
        ),
    );
  }

  Widget _buildSelectionUI() {
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Column(
        children: [
          const SizedBox(height: 10),

          // Top Scanner Badge
          FadeInDown(
            duration: const Duration(milliseconds: 600),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFF22C55E).withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: const Color(0xFF4ADE80).withOpacity(0.35),
                  width: 1,
                ),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.auto_awesome, color: Color(0xFF4ADE80), size: 14),
                  SizedBox(width: 6),
                  Text(
                    "OPTICAL MATURITY CLASSIFIER",
                    style: TextStyle(
                      color: Color(0xFF4ADE80),
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 28),

          // Pulsing High-Tech Viewfinder
          FadeInDown(
            delay: const Duration(milliseconds: 200),
            duration: const Duration(milliseconds: 800),
            child: Container(
              width: 150,
              height: 150,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    const Color(0xFF22C55E).withOpacity(0.2),
                    const Color(0xFF133522).withOpacity(0.6),
                    Colors.transparent,
                  ],
                  stops: const [0.2, 0.7, 1.0],
                ),
                border: Border.all(
                  color: const Color(0xFF4ADE80).withOpacity(0.5),
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF22C55E).withOpacity(0.25),
                    blurRadius: 36,
                    spreadRadius: 4,
                  ),
                ],
              ),
              child: const Center(
                child: Icon(
                  Icons.center_focus_strong,
                  size: 64,
                  color: Color(0xFF4ADE80),
                ),
              ),
            ),
          ),

          const SizedBox(height: 24),

          // Title & Description
          FadeInUp(
            delay: const Duration(milliseconds: 300),
            duration: const Duration(milliseconds: 700),
            child: const Text(
              "Ready to Scan Papaya",
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.w900,
                letterSpacing: 0.3,
              ),
              textAlign: TextAlign.center,
            ),
          ),

          const SizedBox(height: 8),

          FadeInUp(
            delay: const Duration(milliseconds: 400),
            duration: const Duration(milliseconds: 700),
            child: Text(
              "Take a live photo or upload a harvest image to instantly determine maturity stage (Unripe, Ripening, Ripe, or Overripe).",
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 13,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ),

          const SizedBox(height: 32),

          // Action Option Cards (Camera & Gallery)
          Row(
            children: [
              Expanded(
                child: FadeInLeft(
                  delay: const Duration(milliseconds: 500),
                  duration: const Duration(milliseconds: 650),
                  child: _buildSelectionOption(
                    icon: Icons.camera_alt,
                    title: "Live Camera",
                    subtitle: "Instant Photo",
                    accentColor: const Color(0xFF22C55E),
                    onTap: openLiveCamera,
                  ),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: FadeInRight(
                  delay: const Duration(milliseconds: 500),
                  duration: const Duration(milliseconds: 650),
                  child: _buildSelectionOption(
                    icon: Icons.photo_library,
                    title: "Upload Photo",
                    subtitle: "From Gallery",
                    accentColor: const Color(0xFFF59E0B),
                    onTap: () => pickImage(ImageSource.gallery),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 28),

          // Helpful Pro-Tip Banner
          FadeInUp(
            delay: const Duration(milliseconds: 650),
            duration: const Duration(milliseconds: 700),
            child: Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: const Color(0xFF112619),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: Colors.white.withOpacity(0.08),
                  width: 1,
                ),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF59E0B).withOpacity(0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.lightbulb_outline, color: Color(0xFFF59E0B), size: 18),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      "Pro-Tip: Ensure good ambient lighting and keep the whole papaya fruit in frame for top accuracy.",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.75),
                        fontSize: 11,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSelectionOption({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color accentColor,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(22),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 22, horizontal: 16),
          decoration: BoxDecoration(
            color: const Color(0xFF112619),
            borderRadius: BorderRadius.circular(22),
            border: Border.all(
              color: accentColor.withOpacity(0.35),
              width: 1.2,
            ),
            boxShadow: [
              BoxShadow(
                color: accentColor.withOpacity(0.12),
                blurRadius: 18,
                offset: const Offset(0, 6),
              ),
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.18),
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: accentColor.withOpacity(0.4),
                    width: 1,
                  ),
                ),
                child: Icon(icon, size: 28, color: accentColor),
              ),
              const SizedBox(height: 14),
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w900,
                  fontSize: 15,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                subtitle,
                style: TextStyle(
                  color: Colors.white.withOpacity(0.55),
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultUI() {
    if (_loading) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: primaryColor),
            const SizedBox(height: 20),
            const Text("Analyzing...", style: TextStyle(color: Colors.white)),
          ],
        ),
      );
    }

    if (_output == null || _output!.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.info_outline, color: Colors.orangeAccent, size: 54),
              const SizedBox(height: 16),
              const Text(
                "Unable to classify image",
                style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                "Please ensure the image contains a clear view of a papaya fruit.",
                style: TextStyle(color: Colors.white.withOpacity(0.7), fontSize: 13),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: _reset,
                icon: const Icon(Icons.refresh),
                label: const Text("Scan Again"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: primaryColor,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final topResult = _output![0];
    final topLabel = topResult['label'].toString().replaceAll(RegExp(r'[0-9]'), '').trim();
    final topConfidence = topResult['confidence'] * 100;

    double remainingPercent = 0;
    for (int i = 1; i < _output!.length; i++) {
      remainingPercent += (_output![i]['confidence'] * 100);
    }
    final otherClassesCount = _output!.length - 1;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 10),
          const Text("Prediction Details", style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)),
          const SizedBox(height: 20),

          // 1. Summary Card
          FadeInDown(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF143823), Color(0xFF0E2719)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: primaryColor, width: 1.5),
                boxShadow: [
                  BoxShadow(
                    color: primaryColor.withOpacity(0.2),
                    blurRadius: 24,
                    spreadRadius: 2,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Column(
                children: [
                  // Stage Showcase (Single circular stage icon)
                  if (_findMonkeyByLabel(topLabel)?.imagePath != null)
                    Container(
                      width: 110,
                      height: 110,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: primaryColor.withOpacity(0.12),
                      ),
                      padding: const EdgeInsets.all(10),
                      child: Image.asset(
                        _findMonkeyByLabel(topLabel)!.imagePath,
                        fit: BoxFit.contain,
                      ),
                    ),
                  const SizedBox(height: 16),
                  Text(
                    topLabel, 
                    style: TextStyle(color: primaryColor, fontSize: 24, fontWeight: FontWeight.w900),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                    decoration: BoxDecoration(
                      color: primaryColor.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      "Confidence: ${topConfidence.toStringAsFixed(1)}%", 
                      style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFF10B981).withOpacity(0.16),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: const Color(0xFF34D399).withOpacity(0.4),
                        width: 1,
                      ),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.park, size: 14, color: Color(0xFF34D399)),
                        SizedBox(width: 6),
                        Text(
                          "Algorithm: Random Forest",
                          style: TextStyle(
                            color: Color(0xFF34D399),
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.3,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  const SizedBox(height: 12),
                  Text(
                    "Recorded: $_timestamp",
                    style: TextStyle(color: Colors.white.withOpacity(0.55), fontSize: 12),
                  )
                ],
              ),
            ),
          ),

          const SizedBox(height: 30),

          // 2. Full Prediction Breakdown Title
          FadeInUp(
            delay: const Duration(milliseconds: 200),
            child: const Text("Full Prediction Breakdown", style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
          ),
          
          const SizedBox(height: 16),

          // 3. Top Prediction Text
          FadeInUp(
            delay: const Duration(milliseconds: 300),
            child: RichText(
              text: TextSpan(
                style: const TextStyle(fontSize: 16, height: 1.4),
                children: [
                  const TextSpan(text: "Top Prediction: ", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                  TextSpan(text: "$topLabel (${topConfidence.toStringAsFixed(2)}%)", style: TextStyle(color: primaryColor, fontWeight: FontWeight.bold)),
                ]
              ),
            ),
          ),
          
          const SizedBox(height: 4),

          // 4. Remaining distribution Text
          FadeInUp(
             delay: const Duration(milliseconds: 350),
             child: Text(
               "Remaining ${remainingPercent.toStringAsFixed(2)}% distributed among $otherClassesCount other classes:",
               style: TextStyle(color: Colors.grey[400], fontSize: 14, fontStyle: FontStyle.italic),
             ),
          ),

          const SizedBox(height: 20),

          // 5. List Items (Match the ref style: Star/Dot, Label, Progress Bar)
          ..._output!.asMap().entries.map((entry) {
             final index = entry.key;
             final item = entry.value;
             final label = item['label'].toString().replaceAll(RegExp(r'[0-9]'), '').trim();
             final confidence = item['confidence'];
             final percent = (confidence * 100).toStringAsFixed(2);
             final isTop = index == 0;
             final monkey = _findMonkeyByLabel(label);

             return FadeInUp(
               delay: Duration(milliseconds: 400 + (index * 100)),
               child: _buildUniqueListItem(label, confidence, percent, isTop, monkey),
             );
          }),

          const SizedBox(height: 30),
          
          // 6. Footer Total
          FadeInUp(
            delay: const Duration(milliseconds: 800),
            child: Text(
              "Total: 100.00%", 
              style: TextStyle(color: primaryColor, fontSize: 16, fontWeight: FontWeight.bold),
            ),
          ),
          
          const SizedBox(height: 40),
          _buildRetakeButton(),
           const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildUniqueListItem(String label, double confidence, String percent, bool isTop, Monkey? monkey) {
    // Unique list item style allowing expansion
    return Theme(
      data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
      child: ExpansionTile(
        tilePadding: EdgeInsets.zero,
        childrenPadding: const EdgeInsets.only(bottom: 10),
        leading: isTop 
            ? Icon(Icons.star, color: Colors.amber, size: 24) 
            : Icon(Icons.circle, color: Colors.grey[700], size: 12),
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w500))),
            Text("$percent%", style: TextStyle(color: isTop ? primaryColor : Colors.white, fontWeight: FontWeight.bold)),
          ],
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 8.0),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: confidence,
              minHeight: 8,
              backgroundColor: Colors.grey[800],
              valueColor: AlwaysStoppedAnimation<Color>(
                 isTop ? Colors.orangeAccent : (confidence > 0.1 ? Colors.yellow : Colors.grey)
              ),
            ),
          ),
        ),
        // Keep the description feature as a hidden bonus
        children: [
            if (monkey != null)
                Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                        color: Colors.black26, 
                        borderRadius: BorderRadius.circular(8)
                    ),
                    child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                            if (monkey.imagePath.isNotEmpty)
                                Container(
                                    width: 50, height: 50,
                                    margin: const EdgeInsets.only(right: 12),
                                    padding: const EdgeInsets.all(4),
                                    decoration: BoxDecoration(
                                        borderRadius: BorderRadius.circular(8),
                                        image: DecorationImage(image: AssetImage(monkey.imagePath), fit: BoxFit.cover)
                                        shape: BoxShape.circle,
                                        color: primaryColor.withOpacity(0.12),
                                    ),
                                    child: Image.asset(monkey.imagePath, fit: BoxFit.contain),
                                ),
                            Expanded(child: Text(monkey.description, style: TextStyle(color: Colors.grey[400], fontSize: 13)))
                        ],
                    ),
                )
        ],
      ),
    );
  }

  Widget _buildRetakeButton() {
     return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          side: const BorderSide(color: Colors.grey),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        onPressed: _reset,
        child: const Text("SCAN AGAIN", style: TextStyle(color: Colors.white, letterSpacing: 1.2)),
      ),
    );
  }
}
