class Target {
  const Target({
    required this.ticId,
    required this.name,
    required this.ra,
    required this.dec,
    required this.magnitude,
    required this.spectralType,
    this.score,
  });

  final int ticId;
  final String name;
  final double ra;
  final double dec;
  final double magnitude;
  final String spectralType;
  final double? score;

  factory Target.fromJson(Map<String, dynamic> json) => Target(
    ticId: json['tic_id'] as int,
    name: json['name'] as String,
    ra: (json['ra'] as num).toDouble(),
    dec: (json['dec'] as num).toDouble(),
    magnitude: (json['magnitude'] as num).toDouble(),
    spectralType: json['spectral_type'] as String,
    score: (json['score'] as num?)?.toDouble(),
  );
}

class ObservationProduct {
  const ObservationProduct({
    required this.sector,
    required this.mission,
    required this.year,
    required this.exposureSeconds,
    required this.sourceUri,
  });

  final int sector;
  final String mission;
  final int year;
  final double exposureSeconds;
  final String sourceUri;

  factory ObservationProduct.fromJson(Map<String, dynamic> json) =>
      ObservationProduct(
        sector: json['sector'] as int,
        mission: json['mission'] as String,
        year: json['year'] as int,
        exposureSeconds: (json['exposure_seconds'] as num).toDouble(),
        sourceUri: json['source_uri'] as String,
      );
}

class Prediction {
  const Prediction({
    required this.probability,
    required this.label,
    required this.depth,
    required this.durationDays,
    required this.periodDays,
    required this.modelVersion,
    required this.scoreKind,
    required this.dataSource,
    required this.mission,
    required this.sector,
    required this.cadenceCount,
    required this.time,
    required this.flux,
  });

  final double probability;
  final String label;
  final double depth;
  final double durationDays;
  final double periodDays;
  final String modelVersion;
  final String scoreKind;
  final String dataSource;
  final String mission;
  final int sector;
  final int cadenceCount;
  final List<double> time;
  final List<double> flux;

  factory Prediction.fromJson(Map<String, dynamic> json) => Prediction(
    probability: (json['probability'] as num).toDouble(),
    label: json['label'] as String,
    depth: (json['depth'] as num).toDouble(),
    durationDays: (json['duration_days'] as num).toDouble(),
    periodDays: (json['period_days'] as num).toDouble(),
    modelVersion: json['model_version'] as String,
    scoreKind: json['score_kind'] as String,
    dataSource: json['data_source'] as String,
    mission: json['mission'] as String,
    sector: json['sector'] as int,
    cadenceCount: json['cadence_count'] as int,
    time: (json['time'] as List)
        .map((value) => (value as num).toDouble())
        .toList(),
    flux: (json['normalized_flux'] as List)
        .map((value) => (value as num).toDouble())
        .toList(),
  );
}
