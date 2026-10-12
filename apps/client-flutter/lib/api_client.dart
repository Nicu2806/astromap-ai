import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

class ApiClient {
  ApiClient({
    this.baseUrl = const String.fromEnvironment(
      'ASTROMAP_API_URL',
      defaultValue: 'http://127.0.0.1:8080',
    ),
    http.Client? client,
  }) : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Future<List<Target>> targets({double minimumScore = 0}) async {
    final response = await _client.get(
      Uri.parse('$baseUrl/api/v1/targets?min_score=$minimumScore'),
    );
    _requireSuccess(response);
    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return (body['items'] as List)
        .map((item) => Target.fromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<List<ObservationProduct>> observations(int ticId) async {
    final response = await _client.get(
      Uri.parse('$baseUrl/api/v1/targets/$ticId/observations'),
    );
    _requireSuccess(response);
    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return (body['items'] as List)
        .map(
          (item) => ObservationProduct.fromJson(item as Map<String, dynamic>),
        )
        .toList();
  }

  Future<Prediction> analyze(int ticId, {int? sector}) async {
    final query = sector == null ? '' : '?sector=$sector';
    final response = await _client.post(
      Uri.parse('$baseUrl/api/v1/analyses/$ticId$query'),
    );
    _requireSuccess(response);
    final body = jsonDecode(response.body) as Map<String, dynamic>;
    return Prediction.fromJson(body['prediction'] as Map<String, dynamic>);
  }

  void _requireSuccess(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw ApiException('API ${response.statusCode}: ${response.body}');
    }
  }
}

class ApiException implements Exception {
  const ApiException(this.message);
  final String message;

  @override
  String toString() => message;
}
