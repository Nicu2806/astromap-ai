import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'api_client.dart';
import 'models.dart';

class AstroMapApp extends StatelessWidget {
  const AstroMapApp({super.key, this.apiClient});

  final ApiClient? apiClient;

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'AstroMap-AI',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xFF007C7A),
        primary: const Color(0xFF007C7A),
        secondary: const Color(0xFFE05A47),
        surface: const Color(0xFFF6F4EF),
      ),
      scaffoldBackgroundColor: const Color(0xFFECEAE4),
      fontFamily: 'Georgia',
      cardTheme: const CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(6)),
          side: BorderSide(color: Color(0xFFD3D0C8)),
        ),
      ),
    ),
    home: AstroWorkspace(apiClient: apiClient ?? ApiClient()),
  );
}

class AstroWorkspace extends StatefulWidget {
  const AstroWorkspace({super.key, required this.apiClient});

  final ApiClient apiClient;

  @override
  State<AstroWorkspace> createState() => _AstroWorkspaceState();
}

class _AstroWorkspaceState extends State<AstroWorkspace> {
  static const fallbackTargets = [
    Target(
      ticId: 150428135,
      name: 'TOI 700',
      ra: 97.0967865903209,
      dec: -65.5793115481931,
      magnitude: 10.9102,
      spectralType: 'M2V',
    ),
    Target(
      ticId: 307210830,
      name: 'TIC 307210830',
      ra: 124.531756290083,
      dec: -68.3129998725044,
      magnitude: 9.41242,
      spectralType: 'TIC star',
    ),
    Target(
      ticId: 261136679,
      name: 'TIC 261136679',
      ra: 84.2911879979852,
      dec: -80.4691197969941,
      magnitude: 5.1054,
      spectralType: 'TIC star',
    ),
  ];

  List<Target> targets = fallbackTargets;
  Target selected = fallbackTargets.first;
  List<ObservationProduct> observations = const [];
  int? selectedSector;
  Prediction? prediction;
  bool loading = true;
  bool loadingObservations = false;
  bool analyzing = false;
  bool offline = false;
  double minimumScore = 0;
  String? error;
  String? observationError;

  @override
  void initState() {
    super.initState();
    loadTargets();
  }

  Future<void> loadTargets() async {
    setState(() => loading = true);
    try {
      final result = await widget.apiClient.targets(minimumScore: minimumScore);
      if (!mounted) return;
      setState(() {
        targets = result;
        if (targets.isNotEmpty) selected = targets.first;
        offline = false;
      });
      if (targets.isNotEmpty) await loadObservations(selected);
    } catch (_) {
      if (!mounted) return;
      setState(() {
        targets = fallbackTargets
            .where((target) => (target.score ?? 0) >= minimumScore)
            .toList();
        offline = true;
      });
      if (targets.isNotEmpty) await loadObservations(selected);
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  Future<void> loadObservations(Target target) async {
    setState(() {
      loadingObservations = true;
      observations = const [];
      selectedSector = null;
      observationError = null;
    });
    try {
      final result = await widget.apiClient.observations(target.ticId);
      if (!mounted || selected.ticId != target.ticId) return;
      setState(() {
        observations = result;
        selectedSector = result.isEmpty ? null : result.first.sector;
      });
    } catch (exception) {
      if (!mounted || selected.ticId != target.ticId) return;
      setState(() => observationError = exception.toString());
    } finally {
      if (mounted && selected.ticId == target.ticId) {
        setState(() => loadingObservations = false);
      }
    }
  }

  Future<void> analyze() async {
    setState(() {
      analyzing = true;
      prediction = null;
      error = null;
    });
    try {
      final result = await widget.apiClient.analyze(
        selected.ticId,
        sector: selectedSector,
      );
      if (mounted) setState(() => prediction = result);
    } catch (exception) {
      if (mounted) setState(() => error = exception.toString());
    } finally {
      if (mounted) setState(() => analyzing = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      toolbarHeight: 64,
      backgroundColor: const Color(0xFF17212B),
      foregroundColor: Colors.white,
      title: const Row(
        children: [
          Icon(Icons.public, color: Color(0xFF55C9C5)),
          SizedBox(width: 12),
          Text('AstroMap-AI', style: TextStyle(fontWeight: FontWeight.w700)),
          SizedBox(width: 14),
          Text(
            'Candidate Workbench',
            style: TextStyle(fontSize: 13, color: Color(0xFFB8C2C8)),
          ),
        ],
      ),
      actions: [
        Icon(
          offline ? Icons.cloud_off : Icons.cloud_done,
          size: 18,
          color: offline ? const Color(0xFFE9B44C) : const Color(0xFF55C9C5),
        ),
        const SizedBox(width: 7),
        Text(
          offline ? 'DEMO CATALOG' : 'API CONNECTED',
          style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700),
        ),
        const SizedBox(width: 18),
      ],
    ),
    body: LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < 900) {
          return ListView(
            padding: const EdgeInsets.all(12),
            children: [
              SizedBox(height: 340, child: buildSky()),
              const SizedBox(height: 12),
              SizedBox(height: 350, child: buildCatalog()),
              const SizedBox(height: 12),
              SizedBox(height: 680, child: buildInspector()),
            ],
          );
        }
        return Row(
          children: [
            SizedBox(width: 290, child: buildCatalog()),
            const VerticalDivider(width: 1),
            Expanded(flex: 5, child: buildSky()),
            const VerticalDivider(width: 1),
            SizedBox(
              width: math.min(430, constraints.maxWidth * 0.34),
              child: buildInspector(),
            ),
          ],
        );
      },
    ),
  );

  Widget buildCatalog() => ColoredBox(
    color: const Color(0xFFF6F4EF),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 18, 16, 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'TARGET CATALOG',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF647078),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  const Icon(Icons.tune, size: 18),
                  Expanded(
                    child: Slider(
                      value: minimumScore,
                      divisions: 10,
                      label: '${(minimumScore * 100).round()}%',
                      onChanged: (value) =>
                          setState(() => minimumScore = value),
                      onChangeEnd: (_) => loadTargets(),
                    ),
                  ),
                ],
              ),
              Text(
                'Minimum score ${(minimumScore * 100).round()}%',
                style: const TextStyle(fontSize: 12),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        if (loading) const LinearProgressIndicator(minHeight: 2),
        Expanded(
          child: targets.isEmpty
              ? const Center(child: Text('No targets match this filter.'))
              : ListView.separated(
                  itemCount: targets.length,
                  separatorBuilder: (_, _) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final target = targets[index];
                    return ListTile(
                      selected: target.ticId == selected.ticId,
                      selectedTileColor: const Color(0xFFDDEBE8),
                      title: Text(
                        target.name,
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                      subtitle: Text(
                        'TIC ${target.ticId} · ${target.spectralType}',
                      ),
                      trailing: Text(
                        target.score == null
                            ? '—'
                            : '${(target.score! * 100).round()}%',
                      ),
                      onTap: () {
                        setState(() {
                          selected = target;
                          prediction = null;
                          error = null;
                        });
                        loadObservations(target);
                      },
                    );
                  },
                ),
        ),
      ],
    ),
  );

  Widget buildSky() => ColoredBox(
    color: const Color(0xFF07151C),
    child: Stack(
      children: [
        Positioned.fill(
          child: CustomPaint(painter: SkyPainter(targets, selected)),
        ),
        const Positioned(
          left: 18,
          top: 16,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'ICRS ALL-SKY PROJECTION',
                style: TextStyle(
                  color: Color(0xFF9FB4BD),
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),
              SizedBox(height: 4),
              Text(
                'TESS Candidate Field',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
        Positioned(
          right: 14,
          bottom: 14,
          child: Card(
            color: const Color(0xDD17212B),
            child: Padding(
              padding: const EdgeInsets.all(10),
              child: Text(
                'RA ${selected.ra.toStringAsFixed(3)}°  DEC ${selected.dec.toStringAsFixed(3)}°',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ),
      ],
    ),
  );

  Widget buildInspector() => ColoredBox(
    color: Colors.white,
    child: ListView(
      padding: const EdgeInsets.all(18),
      children: [
        Text(
          selected.name,
          style: Theme.of(
            context,
          ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 4),
        Text(
          'TIC ${selected.ticId} · ${selected.spectralType}',
          style: const TextStyle(color: Color(0xFF647078)),
        ),
        const SizedBox(height: 18),
        Row(
          children: [
            Expanded(
              child: Metric('TESS MAG', selected.magnitude.toStringAsFixed(2)),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Metric(
                'CATALOG SCORE',
                selected.score == null
                    ? 'Unchecked'
                    : '${(selected.score! * 100).round()}%',
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        const Text(
          'TESS OBSERVATION',
          style: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w700,
            color: Color(0xFF647078),
          ),
        ),
        const SizedBox(height: 6),
        if (loadingObservations)
          const LinearProgressIndicator(minHeight: 2)
        else if (observationError != null)
          const Notice(
            Icons.cloud_off_outlined,
            'MAST observations are unavailable. Check the backend and network connection.',
          )
        else
          DropdownButtonFormField<int>(
            key: ValueKey('${selected.ticId}-$selectedSector'),
            initialValue: selectedSector,
            decoration: const InputDecoration(
              prefixIcon: Icon(Icons.satellite_alt_outlined),
              contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            ),
            items: observations
                .map(
                  (observation) => DropdownMenuItem(
                    value: observation.sector,
                    child: Text(
                      'Sector ${observation.sector} · ${observation.year} · ${observation.exposureSeconds.round()} s',
                    ),
                  ),
                )
                .toList(),
            onChanged: analyzing
                ? null
                : (sector) => setState(() {
                    selectedSector = sector;
                    prediction = null;
                    error = null;
                  }),
          ),
        const SizedBox(height: 14),
        FilledButton.icon(
          onPressed: analyzing || loadingObservations || selectedSector == null
              ? null
              : analyze,
          icon: analyzing
              ? const SizedBox.square(
                  dimension: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.science_outlined),
          label: Text(
            analyzing ? 'Analyzing photometry…' : 'Run transit analysis',
          ),
        ),
        if (error != null) ...[
          const SizedBox(height: 12),
          const Notice(
            Icons.error_outline,
            'Analysis unavailable. Start Python inference and the Rust API.',
          ),
        ],
        if (prediction == null && error == null) ...[
          const SizedBox(height: 18),
          const Notice(
            Icons.info_outline,
            'Run analysis to generate a normalized light curve and parameter estimate.',
          ),
        ],
        if (prediction case final result?) ...[
          const SizedBox(height: 18),
          Row(
            children: [
              Expanded(
                child: Metric(
                  result.scoreKind == 'bls_screening_score'
                      ? 'BLS SCREENING SCORE'
                      : 'AI SCORE (UNVALIDATED)',
                  '${(result.probability * 100).toStringAsFixed(1)}%',
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Metric(
                  'PERIOD',
                  '${result.periodDays.toStringAsFixed(2)} d',
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: Metric(
                  'DEPTH',
                  '${(result.depth * 1000000).toStringAsFixed(0)} ppm',
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Metric(
                  'DURATION',
                  '${(result.durationDays * 24).toStringAsFixed(2)} h',
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          Metric(
            'OBSERVATION',
            '${result.mission} · Sector ${result.sector} · ${result.cadenceCount} cadences',
          ),
          const SizedBox(height: 8),
          const Text(
            'NORMALIZED LIGHT CURVE',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Color(0xFF647078),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 190,
            child: CustomPaint(painter: LightCurvePainter(result.flux)),
          ),
          const SizedBox(height: 8),
          Text(
            result.modelVersion,
            style: const TextStyle(fontSize: 11, color: Color(0xFF647078)),
          ),
          const SizedBox(height: 4),
          SelectableText(
            result.dataSource,
            style: const TextStyle(fontSize: 10, color: Color(0xFF647078)),
          ),
        ],
      ],
    ),
  );
}

class Metric extends StatelessWidget {
  const Metric(this.label, this.value, {super.key});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: const Color(0xFFF2F0EA),
      borderRadius: BorderRadius.circular(4),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w700,
            color: Color(0xFF647078),
          ),
        ),
        const SizedBox(height: 5),
        Text(
          value,
          style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
        ),
      ],
    ),
  );
}

class Notice extends StatelessWidget {
  const Notice(this.icon, this.text, {super.key});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      border: Border.all(color: const Color(0xFFD3D0C8)),
      borderRadius: BorderRadius.circular(4),
    ),
    child: Row(
      children: [
        Icon(icon, size: 20),
        const SizedBox(width: 10),
        Expanded(child: Text(text, style: const TextStyle(fontSize: 13))),
      ],
    ),
  );
}

class SkyPainter extends CustomPainter {
  SkyPainter(this.targets, this.selected);
  final List<Target> targets;
  final Target selected;

  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()..color = const Color(0xFF24414D);
    for (var index = 1; index < 6; index++) {
      final x = size.width * index / 6;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), grid);
    }
    for (var index = 1; index < 4; index++) {
      final y = size.height * index / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
    final random = math.Random(17);
    final stars = Paint()..color = const Color(0x5579A6B8);
    for (var index = 0; index < 110; index++) {
      canvas.drawCircle(
        Offset(
          random.nextDouble() * size.width,
          random.nextDouble() * size.height,
        ),
        random.nextDouble() * 1.4 + 0.4,
        stars,
      );
    }
    for (final target in targets) {
      final point = Offset(
        target.ra / 360 * size.width,
        (90 - target.dec) / 180 * size.height,
      );
      final active = target.ticId == selected.ticId;
      if (active) {
        canvas.drawCircle(point, 11, Paint()..color = const Color(0x44E9B44C));
      }
      canvas.drawCircle(
        point,
        active ? 5 : 3.5,
        Paint()
          ..color = active ? const Color(0xFFE9B44C) : const Color(0xFF55C9C5),
      );
    }
  }

  @override
  bool shouldRepaint(covariant SkyPainter oldDelegate) =>
      oldDelegate.selected.ticId != selected.ticId ||
      oldDelegate.targets != targets;
}

class LightCurvePainter extends CustomPainter {
  LightCurvePainter(this.flux);
  final List<double> flux;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(
      Offset.zero & size,
      Paint()..color = const Color(0xFFF6F4EF),
    );
    if (flux.length < 2) return;
    final minimum = flux.reduce(math.min);
    final maximum = flux.reduce(math.max);
    final range = math.max(maximum - minimum, 1e-6);
    final path = Path();
    for (var index = 0; index < flux.length; index++) {
      final point = Offset(
        index / (flux.length - 1) * size.width,
        size.height -
            ((flux[index] - minimum) / range * (size.height - 20) + 10),
      );
      if (index == 0) {
        path.moveTo(point.dx, point.dy);
      } else {
        path.lineTo(point.dx, point.dy);
      }
    }
    canvas.drawPath(
      path,
      Paint()
        ..color = const Color(0xFF007C7A)
        ..strokeWidth = 1.6
        ..style = PaintingStyle.stroke,
    );
  }

  @override
  bool shouldRepaint(covariant LightCurvePainter oldDelegate) =>
      oldDelegate.flux != flux;
}
