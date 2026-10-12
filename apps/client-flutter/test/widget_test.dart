// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:astromap_client/dashboard.dart';

void main() {
  testWidgets('renders the scientific workbench', (tester) async {
    tester.view.physicalSize = const Size(1400, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const AstroMapApp());
    await tester.pump();

    expect(find.text('AstroMap-AI'), findsOneWidget);
    expect(find.text('TARGET CATALOG'), findsOneWidget);
    expect(find.text('TESS Candidate Field'), findsOneWidget);
    expect(find.text('Run transit analysis'), findsOneWidget);
  });
}
