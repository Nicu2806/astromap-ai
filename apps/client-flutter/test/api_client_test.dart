import 'package:astromap_client/api_client.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('lists observations and forwards the selected sector', () async {
    final requested = <Uri>[];
    final client = ApiClient(
      baseUrl: 'http://api.test',
      client: MockClient((request) async {
        requested.add(request.url);
        if (request.method == 'GET') {
          return http.Response(
            '{"items":[{"sector":3,"mission":"TESS Sector 03","year":2018,"exposure_seconds":120.0,"source_uri":"mast:TESS/product/s3.fits"}],"total":1}',
            200,
          );
        }
        return http.Response(
          '{"status":"completed","prediction":{"tic_id":150428135,"probability":0.4,"label":"unlikely","depth":0.001,"duration_days":0.1,"period_days":3.2,"model_version":"bls-real-0.1.0","score_kind":"bls_screening_score","data_source":"mast:TESS/product/s3.fits","mission":"TESS/SPOC","sector":3,"cadence_count":1000,"time":[1.0],"normalized_flux":[0.999]}}',
          200,
        );
      }),
    );

    final observations = await client.observations(150428135);
    final prediction = await client.analyze(150428135, sector: 3);

    expect(observations.single.sector, 3);
    expect(prediction.sector, 3);
    expect(requested.last.queryParameters['sector'], '3');
  });
}
