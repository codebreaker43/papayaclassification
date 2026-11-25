import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:test/services/random_forest_service.dart';

void main() {
  test('Verify RandomForestService accuracy across studio and realistic background test sets', () async {
    final Map<String, List<String>> testSuites = {
      'Papaya Unripe': [
        'scratch_test_unripe.png',
        'assets/images/papaya_unripe.png',
        'papaya_curated_dataset/test/unripe/unripe_test_0001.jpg',
        'papaya_curated_dataset/test/unripe/unripe_test_0002.jpg',
        'papaya_curated_dataset/test/unripe/unripe_test_0003.jpg',
      ],
      'Papaya Ripening': [
        'scratch_test_ripening.png',
        'assets/images/papaya_ripening.png',
        'papaya_curated_dataset/test/ripening/ripening_test_0001.jpg',
        'papaya_curated_dataset/test/ripening/ripening_test_0002.jpg',
        'papaya_curated_dataset/test/ripening/ripening_test_0003.jpg',
      ],
      'Papaya Ripe': [
        'scratch_test_ripe.png',
        'assets/images/papaya_ripe.png',
        'papaya_curated_dataset/test/ripe/ripe_test_0001.jpg',
        'papaya_curated_dataset/test/ripe/ripe_test_0002.jpg',
        'papaya_curated_dataset/test/ripe/ripe_test_0003.jpg',
      ],
      'Papaya Overripe': [
        'assets/images/papaya_overripe.png',
        'papaya_curated_dataset/test/overripe/overripe_test_0001.jpg',
        'papaya_curated_dataset/test/overripe/overripe_test_0002.jpg',
        'papaya_curated_dataset/test/overripe/overripe_test_0003.jpg',
      ],
    };

    int total = 0;
    int correct = 0;

    for (final entry in testSuites.entries) {
      final expectedLabel = entry.key;
      for (final filename in entry.value) {
        final file = File(filename);
        if (!file.existsSync()) continue;
        total++;
        final bytes = await file.readAsBytes();
        final result = await RandomForestService.classifyImageBytes(bytes);
        final topLabel = result.first['label'] as String;
        final topConf = (result.first['confidence'] as double) * 100.0;
        final isOk = topLabel == expectedLabel;
        if (isOk) correct++;
        print("[${isOk ? 'OK' : 'FAIL'}] $filename -> Predicted: $topLabel (${topConf.toStringAsFixed(1)}%) | Expected: $expectedLabel");
        expect(topLabel, equals(expectedLabel), reason: "Mismatch on $filename");
      }
    }
    print("Total Dart RandomForestService Accuracy: $correct / $total (${(correct / total * 100).toStringAsFixed(1)}%)");
  });
}
